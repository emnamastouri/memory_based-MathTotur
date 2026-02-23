import os
import re
import base64
import streamlit as st

from core.verify_engine import verify
from core.parse_blocks import extract_blocks, get_final_answer
from core.verify_fix import auto_fix_solution_for_verify

from generator.store_generated import load_generated, append_generated
from generator.data_loader import add_generated_exercise, load_exercises
from generator.retriever import filter_pool, retrieve_similar
from generator.exercise_generator import generate_exercise
from memory.embedder import Embedder
from core.tutor_policy import build_system_prompt
from core.llm_client import LLMClient


DB_PATH = "data/fine-tuning-database.json"
GEN_PATH = "data/generated_exercises.jsonl"
LOGO_PATH = ""

MODEL_CHOICES = {
    "Gemini 3 Flash (rapide)": "gemini-3-flash",
    "Gemini 3 Pro (meilleur raisonnement)": "gemini-3-pro",
    "Gemini 2.5 Flash Lite (économique)": "gemini-2.5-flash-lite",
}

SECTIONS = [
    "Mathématiques",
    "Sciences de l'informatique",
    "Sciences Techniques",
    "Sport",
    "Sciences Expérimentales",
    "Économie et Gestion",
]

st.set_page_config(page_title="MathTutorAI — Générateur d'exercices", layout="wide")


# ----------------------------
# UI: Background logo (flou)
# ----------------------------
def inject_blurred_logo_background(logo_path: str):
    if not logo_path or not os.path.exists(logo_path):
        return
    try:
        with open(logo_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")

        css = f"""
        <style>
        .stApp {{
            position: relative;
        }}
        .stApp::before {{
            content: "";
            position: fixed;
            inset: 0;
            background-image: url("data:image/png;base64,{b64}");
            background-repeat: no-repeat;
            background-position: center;
            background-size: min(620px, 70vw);
            opacity: 0.30;
            filter: blur(1px);
            transform: scale(1.05);
            z-index: 0;
            pointer-events: none;
        }}
        .stApp > header, .stApp > div {{
            position: relative;
            z-index: 1;
        }}
        div[data-testid="stSidebar"] {{
            background: rgba(15, 23, 42, 0.45);
            backdrop-filter: blur(6px);
        }}
        </style>
        """
        st.markdown(css, unsafe_allow_html=True)
    except Exception:
        return


inject_blurred_logo_background(LOGO_PATH)

st.title("MathTutorAI — Génération d'exercices (Bac Tunisie)")
st.caption("Interface en français • Sélectionnez une section et un chapitre, puis générez un exercice.")


# ----------------------------
# Helpers
# ----------------------------
def get_api_key() -> str:
    k = os.getenv("GEMINI_API_KEY", "").strip()
    if k:
        return k
    try:
        return str(st.secrets.get("GEMINI_API_KEY", "")).strip()
    except Exception:
        return ""


def get_llm(model_name: str) -> LLMClient:
    api_key = get_api_key()
    if not api_key:
        st.error(
            "Clé API Gemini introuvable.\n\n"
            "✅ Solutions :\n"
            "1) Définir la variable d'environnement GEMINI_API_KEY\n"
            "ou\n"
            "2) Créer .streamlit/secrets.toml avec GEMINI_API_KEY"
        )
        st.stop()
    return LLMClient(api_key=api_key, model_name=model_name)


def parse_exercise_solution(gen_text: str) -> tuple[str, str]:
    """Parse raw generation text into (exercise, solution). Solution may be empty."""
    txt = (gen_text or "").strip().replace("\r\n", "\n")
    if not txt:
        return "", ""

    patterns = [
        r"\nSOLUTION\s*:\s*",
        r"\nSolution\s*:\s*",
        r"\nCORRECTION\s*:\s*",
        r"\nCorrection\s*:\s*",
    ]

    split_pos = None
    sol_start = None
    for p in patterns:
        m = re.search(p, txt)
        if m:
            split_pos = m.start()
            sol_start = m.end()
            break

    if split_pos is not None and sol_start is not None:
        ex_part = txt[:split_pos]
        sol = txt[sol_start:].strip()
        ex = re.sub(r"^EXERCICE\s*:\s*", "", ex_part.strip(), flags=re.IGNORECASE)
        return ex, sol

    ex = re.sub(r"^EXERCICE\s*:\s*", "", txt.strip(), flags=re.IGNORECASE)
    return ex, ""


def build_solution_blocks(enonce: str, raw_solution: str) -> str:
    """
    Ensure strict blocks:
    EXERCICE:
    SOLUTION:
    FINAL_ANSWER:
    CHECK:
    """
    enonce = (enonce or "").strip()
    raw_solution = (raw_solution or "").strip()

    blocks = extract_blocks(raw_solution)
    if blocks and ("FINAL_ANSWER" in blocks or "CHECK" in blocks or "SOLUTION" in blocks):
        if "EXERCICE" not in blocks:
            blocks["EXERCICE"] = enonce
        parts = []
        for h in ["EXERCICE", "SOLUTION", "FINAL_ANSWER", "CHECK"]:
            if h in blocks and (blocks[h] or "").strip():
                parts.append(f"{h}:\n{blocks[h].strip()}")
        return "\n\n".join(parts).strip()

    return (
        f"EXERCICE:\n{enonce}\n\n"
        f"SOLUTION:\n{raw_solution}\n\n"
        f"FINAL_ANSWER:\n\n"
        f"CHECK:\n"
    ).strip()


def render_verification(rep):
    st.markdown("### ✅ Vérification SymPy / Plugins")
    if getattr(rep, "ok", False):
        st.success(getattr(rep, "summary", "✔ Vérification OK"))
    else:
        st.error(getattr(rep, "summary", "❌ Problèmes détectés"))

    with st.expander("Voir les détails de la vérification (SymPy)"):
        if hasattr(rep, "items"):
            for it in rep.items:
                st.write(("✅" if it.ok else "❌"), it.name, "—", it.message)


@st.cache_data(show_spinner=False)
def load_db():
    base = load_exercises(DB_PATH)
    gen = load_generated(GEN_PATH)

    from generator.data_loader import Exercise

    gen_ex = []
    for i, r in enumerate(gen):
        gen_ex.append(
            Exercise(
                ex_id=str(r.get("id", f"gen_{i}")),
                grade=str(r.get("grade", "")),
                section=str(r.get("section", "")),
                topic=str(r.get("topic", "")),
                enonce=str(r.get("enonce", "")),
                solution=str(r.get("solution", "")),
                metadata=r.get("metadata", {}),
            )
        )
    return base + gen_ex


# ----------------------------
# Data + Sidebar UI (FR)
# ----------------------------
exercises = load_db()

with st.sidebar:
    st.header("🎯 Sélection")

    section = st.selectbox("Section", SECTIONS, index=0)

    topics = sorted(set(e.topic for e in exercises if e.section == section))
    if not topics:
        st.error("Aucun chapitre trouvé pour cette section dans la base JSON.")
        st.stop()

    topic = st.selectbox("Chapitre / Thème", topics, index=0)

    st.divider()
    st.header("⚙️ Paramètres")

    model_label = st.selectbox("Modèle Gemini", options=list(MODEL_CHOICES.keys()), index=0)
    model_name = MODEL_CHOICES[model_label]

    top_k = st.slider("Nombre d'exemples similaires (Top-K)", 1, 6, 3)

st.subheader("1) Récupération d'exemples similaires (mémoire Bac Tunisie)")
pool = filter_pool(exercises, section, topic)
st.write(f"Taille du pool filtré : **{len(pool)}** exercices")

embedder = Embedder()


# ----------------------------
# Session state
# ----------------------------
st.session_state.setdefault("exercise_enonce", "")

# IMPORTANT:
# - hidden_solution: solution (ou correction) stockée mais NON affichée tant que le bouton n'est pas cliqué.
# - exercise_solution: ce qui est affiché à l'écran (solution visible).
st.session_state.setdefault("hidden_solution", "")
st.session_state.setdefault("exercise_solution", "")

st.session_state.setdefault("solution_blocks", "")
st.session_state.setdefault("verified_solution_blocks", "")
st.session_state.setdefault("chat_history", [])
st.session_state.setdefault("verify_report", None)
st.session_state.setdefault("saved_current", False)
st.session_state.setdefault("retrieved", [])
st.session_state.setdefault("generated_text", "")

# Flag d'affichage
st.session_state.setdefault("show_solution", False)


# ----------------------------
# UI: Generate exercise
# ----------------------------
colA, colB = st.columns(2)

with colA:
    if st.button("🧠 Générer un nouvel exercice"):
        # Reset everything for a fresh start
        st.session_state.saved_current = False
        st.session_state.verify_report = None
        st.session_state.verified_solution_blocks = ""
        st.session_state.solution_blocks = ""
        st.session_state.exercise_solution = ""
        st.session_state.hidden_solution = ""
        st.session_state.chat_history = []
        st.session_state.generated_text = ""
        st.session_state.show_solution = False  # <-- FIX: hide solution until user clicks

        retrieved = retrieve_similar(embedder=embedder, pool=pool, k=top_k)
        st.session_state.retrieved = retrieved

        llm = get_llm(model_name)

        gen_system_prompt = (
            "Tu es un générateur d'exercices pour le Baccalauréat Tunisien. "
            "Retourne UNIQUEMENT le format demandé."
        )

        gen_text = generate_exercise(
            llm=llm,
            system_prompt=gen_system_prompt,
            section=section,
            topic=topic,
            retrieved=retrieved,
        )

        st.session_state.generated_text = gen_text
        ex, sol = parse_exercise_solution(gen_text)

        st.session_state.exercise_enonce = ex

        # CRUCIAL FIX:
        # On stocke la solution générée en "hidden_solution" mais on n'affiche rien
        st.session_state.hidden_solution = sol  # peut être vide
        st.session_state.exercise_solution = ""  # rien à afficher avant le clic

        # On garde des blocks internes (utile pour debug), mais NON affichés tant que show_solution=False
        st.session_state.solution_blocks = build_solution_blocks(ex, sol)

with colB:
    st.markdown("### 📚 Exemples récupérés")
    retrieved = st.session_state.get("retrieved", [])
    if not retrieved:
        st.info("Cliquez sur **Générer un nouvel exercice** pour récupérer des exemples puis générer.")
    else:
        for r in retrieved:
            st.write(f"**{r.exercise.ex_id}** | score={r.score:.3f}")
            st.write(r.exercise.enonce[:300] + ("..." if len(r.exercise.enonce) > 300 else ""))
            st.divider()

st.divider()
st.subheader("2) Exercice généré")

if st.session_state.exercise_enonce:
    st.markdown("### EXERCICE")
    st.write(st.session_state.exercise_enonce)
else:
    st.info("Aucun exercice généré pour le moment.")

st.divider()
st.subheader("3) Choisir : Solution complète OU Discussion avec le tuteur")

col1, col2 = st.columns(2)

# ----------------------------
# Solution + Verification
# ----------------------------
with col1:
    if st.button("✅ Afficher la solution complète"):
        if not st.session_state.exercise_enonce.strip():
            st.warning("Veuillez d'abord générer un exercice.")
        else:
            # Le clic DOIT activer l'affichage
            st.session_state.show_solution = True

            llm = get_llm(model_name)

            sol_prompt = f"""
Tu es un professeur de mathématiques.

Tu dois répondre STRICTEMENT avec 4 blocs EXACTS dans cet ordre.
Règles CRITIQUES:
- Les titres doivent être seuls sur leur ligne, exactement comme:
  EXERCICE:
  SOLUTION:
  FINAL_ANSWER:
  CHECK:
- FINAL_ANSWER doit être court et parsable par SymPy (nombre, expression, dict, liste...).
- CHECK doit être au format attendu par les verifiers:
  - Équations / système: SYSTEM; Eq(...)
  - Dérivée: DERIVATIVE; var=x; func=<expression>
  - Intégrale: INTEGRAL; var=x; integrand=<expression>
  - Limite: LIMIT; var=x; expr=<expression>; point=<valeur>

EXERCICE:
{st.session_state.exercise_enonce}

SOLUTION:
(donne une solution détaillée et pédagogique)

FINAL_ANSWER:
(donne seulement la réponse finale)

CHECK:
(donne une seule ligne de check)
""".strip()

            out_text = llm.generate(
                system_prompt="Retourne uniquement les 4 blocs demandés, format strict.",
                context="",
                user_prompt=sol_prompt,
            )
            out_text = (out_text or "").replace("\r\n", "\n").strip()

            # Store full blocks
            st.session_state.solution_blocks = out_text
            blocks = extract_blocks(out_text)

            # Store both:
            # - hidden_solution for tutor reference
            # - exercise_solution for display (because now show_solution=True)
            st.session_state.hidden_solution = (blocks.get("SOLUTION", "") or "").strip()
            st.session_state.exercise_solution = st.session_state.hidden_solution

            # ✅ Auto-fix + verify
            fixed_enonce, fixed_blocks = auto_fix_solution_for_verify(
                st.session_state.exercise_enonce,
                st.session_state.solution_blocks,
            )

            rep = verify(topic, fixed_enonce, fixed_blocks)
            st.session_state.verify_report = rep
            st.session_state.verified_solution_blocks = fixed_blocks

    # DISPLAY: only if user has clicked (show_solution=True)
    if st.session_state.show_solution and st.session_state.exercise_solution.strip():
        st.markdown("### SOLUTION")
        st.write(st.session_state.exercise_solution)
    else:
        st.info("Cliquez sur **Afficher la solution complète** pour générer la correction.")

    # Always show verification result if available
    if st.session_state.get("verify_report") is not None:
        render_verification(st.session_state.verify_report)

        with st.expander("Voir le texte exact vérifié (après auto-fix)"):
            st.code(st.session_state.get("verified_solution_blocks", ""), language="text")

    # Save after verification (only once)
    if st.session_state.exercise_enonce.strip() and st.session_state.hidden_solution.strip():
        if st.session_state.get("verify_report") is not None and (not st.session_state.saved_current):
            fa = get_final_answer(st.session_state.get("verified_solution_blocks", "")) or ""

            append_generated(
                GEN_PATH,
                {
                    "id": f"gen_{len(load_generated(GEN_PATH)) + 1}",
                    "section": section,
                    "topic": topic,
                    "enonce": st.session_state.exercise_enonce,
                    "solution": st.session_state.get("verified_solution_blocks", st.session_state.solution_blocks),
                    "final_answer": fa,
                    "model": model_name,
                },
            )

            add_generated_exercise(
                DB_PATH,
                {
                    "section": section,
                    "topic": topic,
                    "exercise": st.session_state.exercise_enonce,
                    "solution": st.session_state.get("verified_solution_blocks", st.session_state.solution_blocks),
                    "final_answer": fa,
                    "tags": [],
                    "model": model_name,
                },
            )

            if hasattr(embedder, "add_document"):
                try:
                    embedder.add_document(
                        text=st.session_state.exercise_enonce + "\n" + st.session_state.hidden_solution,
                        metadata={"section": section, "topic": topic, "model": model_name},
                    )
                except Exception:
                    pass

            st.session_state.saved_current = True
            st.cache_data.clear()
            st.success("Exercice enregistré dans la mémoire ✅")


# ----------------------------
# Tutor chat
# ----------------------------
with col2:
    st.markdown("### 💬 Discussion avec le tuteur (scaffolding)")
    if not st.session_state.exercise_enonce.strip():
        st.info("Générez d'abord un exercice.")
    else:
        for role, msg in st.session_state.chat_history:
            st.chat_message(role).write(msg)

        user_msg = st.chat_input("Posez une question sur l'exercice généré…")
        if user_msg:
            st.session_state.chat_history.append(("user", user_msg))

            llm = get_llm(model_name)
            system_prompt = build_system_prompt()

            # IMPORTANT:
            # Utiliser hidden_solution (même si l'utilisateur n'a pas affiché la solution),
            # pour que le tuteur puisse aider correctement.
            tutor_user_prompt = (
                f"Exercice:\n{st.session_state.exercise_enonce}\n\n"
                f"Référence (solution complète, ne pas tout révéler d'un coup):\n{st.session_state.hidden_solution}\n\n"
                f"Question de l'élève:\n{user_msg}\n\n"
                "Règle: répondre en mode tuteur (indices d'abord, questions guidées). "
                "Ne donne la solution complète que si l'élève la demande explicitement."
            )

            out_text = llm.generate(system_prompt=system_prompt, context="", user_prompt=tutor_user_prompt)
            st.session_state.chat_history.append(("assistant", out_text))

            st.rerun()