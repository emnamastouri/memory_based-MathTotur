import os
import re
import base64
import streamlit as st
from core.verify_engine import verify
from generator.store_generated import load_generated, append_generated
from core.auto_verify import verify_by_topic
from core.verify_engine import verify
from generator.data_loader import add_generated_exercise, load_exercises
from generator.retriever import filter_pool, retrieve_similar
from generator.exercise_generator import generate_exercise
from memory.embedder import Embedder
from core.tutor_policy import build_system_prompt
from core.llm_client import LLMClient

DB_PATH  = "data/fine-tuning-database.json"
GEN_PATH = "data/generated_exercises.jsonl"
LOGO_PATH = "assets/logo.png"
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
    """
    Adds a blurred, semi-transparent background logo.
    Works best if the logo is a PNG with transparent background.
    """
    if not os.path.exists(logo_path):
        return

    try:
        with open(logo_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")

        css = f"""
        <style>
        /* Put a blurred logo behind everything */
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
            opacity: 0.30;           /* adjust visibility */
            filter: blur(1px);       /* blur amount */
            transform: scale(1.05);
            z-index: 0;
            pointer-events: none;
        }}

        /* Make content appear above background */
        .stApp > header, .stApp > div {{
            position: relative;
            z-index: 1;
        }}

        /* Slightly improve readability on top of background */
        div[data-testid="stSidebar"] {{
            background: rgba(15, 23, 42, 0.45);
            backdrop-filter: blur(6px);
        }}
        </style>
        """
        st.markdown(css, unsafe_allow_html=True)
    except Exception:
        # If anything fails, just skip background
        return


inject_blurred_logo_background(LOGO_PATH)

st.title("MathTutorAI — Génération d'exercices (Bac Tunisie)")
st.caption("Interface en français • Sélectionnez une section et un chapitre, puis générez un exercice.")

# ----------------------------
# Helpers
# ----------------------------
def get_api_key() -> str:
    # 1) env var (never crashes)
    k = os.getenv("GEMINI_API_KEY", "").strip()  or st.secrets["GEMINI_API_KEY"]
    if k:
        return k
    # 2) Streamlit secrets (can raise if no secrets file exists)
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


@st.cache_data(show_spinner=False)
def load_db():
    base = load_exercises(DB_PATH)  # list[Exercise]
    gen = load_generated(GEN_PATH)  # list[dict]

    # Convert generated dicts to Exercise-like objects (minimal)
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

    model_label = st.selectbox(
        "Modèle Gemini",
        options=list(MODEL_CHOICES.keys()),
        index=0,
    )
    model_name = MODEL_CHOICES[model_label]

    top_k = st.slider("Nombre d'exemples similaires (Top-K)", 1, 6, 3)

st.subheader("1) Récupération d'exemples similaires (mémoire Bac Tunisie)")
pool = filter_pool(exercises, section, topic)
st.write(f"Taille du pool filtré : **{len(pool)}** exercices")

embedder = Embedder()

# session state init
st.session_state.setdefault("retrieved", [])
st.session_state.setdefault("generated_text", "")
st.session_state.setdefault("exercise_enonce", "")
st.session_state.setdefault("exercise_solution", "")
st.session_state.setdefault("chat_history", [])
st.session_state.setdefault("verify_report", None)
st.session_state.setdefault("saved_current", False)  # prevent double-saving on reruns

colA, colB = st.columns(2)

with colA:
    if st.button("🧠 Générer un nouvel exercice"):
        st.session_state.saved_current = False  # new exercise -> not saved yet

        retrieved = retrieve_similar(embedder=embedder, pool=pool, k=top_k)
        st.session_state.retrieved = retrieved

        llm = get_llm(model_name)

        # IMPORTANT: generator system prompt (not tutor prompt)
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

        # store raw + parsed
        st.session_state.generated_text = gen_text
        ex, sol = parse_exercise_solution(gen_text)
        st.session_state.exercise_enonce = ex
        st.session_state.exercise_solution = sol
        st.session_state.chat_history = []  # reset chat for new exercise

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

with col1:
    if st.button("✅ Afficher la solution complète"):
        if not st.session_state.exercise_enonce.strip():
            st.warning("Veuillez d'abord générer un exercice.")
        else:
            # If solution missing, generate it on demand
            if not st.session_state.exercise_solution.strip():
                llm = get_llm(model_name)

                sol_prompt = (
                    "Donne la SOLUTION complète et détaillée (étape par étape) de l'exercice suivant.\n\n"
                    f"EXERCICE:\n{st.session_state.exercise_enonce}\n\n"
                    "Réponds avec le format:\nSOLUTION:\n<...>"
                )

                out_text = llm.generate(
                    system_prompt="Tu es un professeur de mathématiques. Retourne uniquement la solution.",
                    context="",
                    user_prompt=sol_prompt,
                )

                s = (out_text or "").strip()
                # parse after "SOLUTION:" if present
                report = verify_by_topic(topic, st.session_state.exercise_enonce, st.session_state.exercise_solution)
                st.session_state.verify_report = report

                if "SOLUTION" in s.upper() and ":" in s:
                    st.session_state.exercise_solution = s.split(":", 1)[-1].strip()
                else:
                    st.session_state.exercise_solution = s

            st.markdown("### SOLUTION")
            st.write(st.session_state.exercise_solution)

            # Save only once for this exercise (avoid duplicates on reruns)
            if not st.session_state.saved_current:
                # 1) Save to generated_exercises.jsonl
                append_generated(
                    GEN_PATH,
                    {
                        "id": f"gen_{len(load_generated(GEN_PATH)) + 1}",
                        "section": section,
                        "topic": topic,
                        "enonce": st.session_state.exercise_enonce,
                        "solution": st.session_state.exercise_solution,
                        "model": model_name,
                    },
                )

                # 2) Save to fine-tuning database
                add_generated_exercise(
                    DB_PATH,
                    {
                        "section": section,
                        "topic": topic,
                        "exercise": st.session_state.exercise_enonce,
                        "solution": st.session_state.exercise_solution,
                        "final_answer": "",
                        "tags": [],
                        "model": model_name,
                    },
                )

                # 3) Update embedder memory IF supported (won't crash if not)
                if hasattr(embedder, "add_document"):
                    try:
                        embedder.add_document(
                            text=st.session_state.exercise_enonce + "\n" + st.session_state.exercise_solution,
                            metadata={"section": section, "topic": topic, "model": model_name},
                        )
                    except Exception:
                        pass

                st.session_state.saved_current = True
                st.cache_data.clear()  # so next run includes new items
                st.success("Exercice enregistré dans la mémoire ✅")

                report = verify(topic, st.session_state.exercise_enonce, st.session_state.exercise_solution)
                st.session_state.verify_report = report
                if report.ok:
                    st.success(report.summary)
                else:
                    st.warning(report.summary)

                with st.expander("Détails vérification"):
                 for it in report.items:
                  st.write(("✅" if it.ok else "❌"), it.name, "—", it.message)
                  st.json(report.details)

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

            tutor_user_prompt = (
                f"Exercice:\n{st.session_state.exercise_enonce}\n\n"
                f"Référence (solution complète, ne pas tout révéler d'un coup):\n{st.session_state.exercise_solution}\n\n"
                f"Question de l'élève:\n{user_msg}\n\n"
                "Règle: répondre en mode tuteur (indices d'abord, questions guidées). "
                "Ne donne la solution complète que si l'élève la demande explicitement."
            )

            out_text = llm.generate(system_prompt=system_prompt, context="", user_prompt=tutor_user_prompt)
            st.session_state.chat_history.append(("assistant", out_text))
            if "FINAL_ANSWER" in (out_text or "") or "CHECK" in (out_text or ""):


                report = verify(topic, st.session_state.exercise_enonce, st.session_state.exercise_solution)
                st.session_state.verify_report = report
                if report.ok:
                    st.success(report.summary)
                else:
                    st.warning(report.summary)

                with st.expander("Détails vérification"):
                 for it in report.items:
                  st.write(("✅" if it.ok else "❌"), it.name, "—", it.message)
                  st.json(report.details)

            st.rerun()
