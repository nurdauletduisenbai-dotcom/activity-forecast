"""Streamlit қосымшасы: келесі аптаның оқу белсенділігін болжау.
Іске қосу: streamlit run app.py
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

BASE = Path(__file__).parent
MODEL_PATH = BASE / "models" / "activity_model.joblib"
META_PATH = BASE / "models" / "model_metadata.json"
METRICS = ["logins", "active_minutes", "video_views", "tasks_submitted", "forum_posts", "quiz_score"]

st.set_page_config(page_title="Оқу белсенділігінің болжамы", page_icon="📈", layout="wide")


@st.cache_resource
def load_model():
    model = joblib.load(MODEL_PATH)
    meta = json.loads(META_PATH.read_text(encoding="utf-8"))
    return model, meta


def build_row(prev: dict, cur: dict, week_t: int, gpa: float, next_deadline: bool, next_holiday: bool) -> dict:
    """Notebook-тегі make_features() функциясымен бірдей белгілер."""
    row = {"week_t": week_t, "prior_gpa": gpa}
    for m in METRICS:
        row[f"{m}_w0"] = cur[m]
        row[f"{m}_w1"] = prev[m]
    row["minutes_delta"] = cur["active_minutes"] - prev["active_minutes"]
    row["minutes_mean2"] = (cur["active_minutes"] + prev["active_minutes"]) / 2
    row["next_is_deadline"] = int(next_deadline)
    row["next_is_holiday"] = int(next_holiday)
    return row


def risk_label(pred: float, cutoff: float) -> str:
    return "🔴 Қауіп: белсенділік төмендеуі мүмкін" if pred < cutoff else "🟢 Қалыпты"


model, meta = load_model()
FEATURES = meta["features"]
CUTOFF = float(meta.get("risk_cutoff_on_prediction", meta["risk_threshold_minutes"]))

st.title("📈 Келесі аптаның оқу белсенділігін болжау")
st.caption("Зерттеу сұрағы: алдыңғы екі аптадағы деректер бойынша келесі аптаның оқу белсенділігін болжауға бола ма?")

with st.sidebar:
    st.header("Модель туралы")
    st.write(f"**Алгоритм:** {meta['model']}")
    tm = meta["test_metrics"]
    st.metric("Тест MAE, минут", f"{tm['MAE']:.1f}",
              delta=f"{tm['MAE'] - meta['baseline_test_MAE']:.1f} базалықпен салыстырғанда", delta_color="inverse")
    st.metric("Тест R²", f"{tm['R2']:.2f}")
    st.write(f"**Қауіп белгісі:** болжам < {CUTOFF:.0f} мин (нақты шек — {meta['risk_threshold_minutes']} мин)")
    st.info("Болжам — оқытушыға арналған көмекші сигнал, білім алушыны бағалау құралы емес.")

tab1, tab2 = st.tabs(["👤 Бір білім алушы", "📄 Топ бойынша (CSV)"])

with tab1:
    c0, c1, c2 = st.columns([1, 1.2, 1.2])
    with c0:
        st.subheader("Жалпы")
        week_t = st.slider("Ағымдағы апта (t)", 2, 15, 8)
        gpa = st.number_input("Алдыңғы GPA", 1.5, 4.0, 3.0, 0.05)
        next_deadline = st.checkbox("Келесі аптада дедлайн бар", value=(week_t + 1) in meta["deadline_weeks"])
        next_holiday = st.checkbox("Келесі апта — демалыс", value=(week_t + 1) in meta["holiday_weeks"])

    def week_inputs(col, title, key, defaults):
        with col:
            st.subheader(title)
            return {
                "logins": st.number_input("LMS-ке кіру саны", 0, 100, defaults[0], key=f"l{key}"),
                "active_minutes": st.number_input("Белсенді минуттар", 0.0, 900.0, defaults[1], 5.0, key=f"m{key}"),
                "video_views": st.number_input("Бейне қарау", 0, 50, defaults[2], key=f"v{key}"),
                "tasks_submitted": st.number_input("Тапсырылған тапсырма (0–3)", 0, 3, defaults[3], key=f"t{key}"),
                "forum_posts": st.number_input("Форум хабарламалары", 0, 30, defaults[4], key=f"f{key}"),
                "quiz_score": st.number_input("Тест балы (0–100)", 0.0, 100.0, defaults[5], 1.0, key=f"q{key}"),
            }

    prev = week_inputs(c1, f"{week_t - 1}-апта (t−1)", "p", (6, 160.0, 4, 3, 1, 65.0))
    cur = week_inputs(c2, f"{week_t}-апта (t)", "c", (4, 110.0, 2, 2, 0, 55.0))

    row = build_row(prev, cur, week_t, gpa, next_deadline, next_holiday)
    pred = float(model.predict(pd.DataFrame([row])[FEATURES])[0])
    pred = max(pred, 0.0)

    st.divider()
    r1, r2, r3 = st.columns(3)
    r1.metric(f"{week_t + 1}-аптаның болжамы", f"{pred:.0f} мин",
              delta=f"{pred - cur['active_minutes']:+.0f} мин ағымдағы аптамен салыстырғанда")
    r2.metric("Мүмкін аралық (±MAE)", f"{max(pred - tm['MAE'], 0):.0f}–{pred + tm['MAE']:.0f} мин")
    r3.markdown(f"### {risk_label(pred, CUTOFF)}")
    chart = pd.DataFrame({"минут": [prev["active_minutes"], cur["active_minutes"], pred]},
                         index=[f"{week_t - 1} (нақты)", f"{week_t} (нақты)", f"{week_t + 1} (болжам)"])
    st.bar_chart(chart)

with tab2:
    st.write("Апталық LMS журналын жүктеңіз (бағандар: `student_id, week, prior_gpa, logins, active_minutes, "
             "video_views, tasks_submitted, forum_posts, quiz_score`). Әр білім алушы үшін соңғы екі апта алынады.")
    up = st.file_uploader("CSV файл", type="csv")
    use_demo = st.button("Демо деректермен көру")
    if up is not None or use_demo:
        df = pd.read_csv(up) if up is not None else pd.read_csv(BASE / "data" / "clean_weekly_activity.csv").query("week <= 13")
        missing = set(["student_id", "week", "prior_gpa", *METRICS]) - set(df.columns)
        if missing:
            st.error(f"Бағандар жетіспейді: {', '.join(sorted(missing))}")
        else:
            df = df.drop_duplicates().sort_values(["student_id", "week"])
            for col in METRICS:
                df[col] = df[col].fillna(df.groupby("student_id")[col].transform("median"))
            t = int(df["week"].max())
            rows, ids = [], []
            for sid, g in df.groupby("student_id"):
                g = g.set_index("week")
                if t in g.index and (t - 1) in g.index:
                    rows.append(build_row(g.loc[t - 1, METRICS].to_dict(), g.loc[t, METRICS].to_dict(), t,
                                          float(g.loc[t, "prior_gpa"]), (t + 1) in meta["deadline_weeks"],
                                          (t + 1) in meta["holiday_weeks"]))
                    ids.append(sid)
            if not rows:
                st.warning("Екі қатар апта деректері бар білім алушы табылмады.")
            else:
                X = pd.DataFrame(rows)[FEATURES]
                res = pd.DataFrame({"student_id": ids,
                                    f"{t}-апта, мин": X["active_minutes_w0"].round(0),
                                    f"{t + 1}-апта болжамы, мин": np.clip(model.predict(X), 0, None).round(0)})
                res["қауіп"] = res[f"{t + 1}-апта болжамы, мин"] < CUTOFF
                res = res.sort_values(f"{t + 1}-апта болжамы, мин")
                st.write(f"**{t}-апта** деректері бойынша **{t + 1}-аптаның** болжамы. "
                         f"Қауіп тобында: **{int(res['қауіп'].sum())}** / {len(res)} білім алушы.")
                st.dataframe(res, width="stretch", hide_index=True)
                st.download_button("Нәтижені CSV ретінде жүктеу", res.to_csv(index=False).encode("utf-8"),
                                   file_name=f"forecast_week_{t + 1}.csv", mime="text/csv")
