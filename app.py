import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Churn Prediction", page_icon="📉", layout="wide")

@st.cache_resource
def load_artifact():
    return joblib.load("xgb_model.joblib")

artifact = load_artifact()
model = artifact["model"]
columns = artifact["columns"]
default_threshold = float(artifact.get("threshold", 0.5))

YES_NO = ["No", "Yes"]
STEPS = ["Profile", "Services", "Contract & billing"]
RESULT_STEP = len(STEPS)

INTERNET_ADDONS = [
    ("security", "Online security"),
    ("backup", "Online backup"),
    ("protection", "Device protection"),
    ("tech", "Tech support"),
    ("tv", "Streaming TV"),
    ("movies", "Streaming movies"),
]

DEFAULTS = {
    "gender": "Female",
    "senior": "No",
    "partner": "No",
    "dependents": "No",
    "tenure": 12,
    "phone": "Yes",
    "multiple": "No",
    "internet": "DSL",
    "security": "No",
    "backup": "No",
    "protection": "No",
    "tech": "No",
    "tv": "No",
    "movies": "No",
    "contract": "Month-to-month",
    "paperless": "Yes",
    "payment": "Electronic check",
    "monthly": 70.0,
    "manual_total": False,
    "total": None,
}

# ----------------------------------------------------------------------------
# Session state: keeps the answers while the user moves between steps
# ----------------------------------------------------------------------------
if "step" not in st.session_state:
    st.session_state.step = 0
if "data" not in st.session_state:
    st.session_state.data = DEFAULTS.copy()

d = st.session_state.data
step = st.session_state.step

def go_to(target: int):
    st.session_state.step = target

def restart():
    st.session_state.data = DEFAULTS.copy()
    st.session_state.step = 0
    for k in list(st.session_state.keys()):
        if k.startswith("w_"):
            del st.session_state[k]

def select(label, key, options):
    current = d[key] if d[key] in options else options[0]
    value = st.selectbox(label, options, index=options.index(current), key=f"w_{key}")
    d[key] = value
    return value

def render_stepper(current: int):
    cols = st.columns(len(STEPS))
    for i, (col, name) in enumerate(zip(cols, STEPS)):
        if i < current:
            col.markdown(f"✅ **{name}**")
        elif i == current:
            col.markdown(f"🔵 **{name}**")
        else:
            col.markdown(f"⚪ {name}")
    st.progress(min((current + 1) / len(STEPS), 1.0))

def get_total_charges() -> float:
    if d["manual_total"] and d["total"] is not None:
        return float(d["total"])
    return round(d["monthly"] * d["tenure"], 2)

def build_input() -> pd.DataFrame:
    row = pd.DataFrame(
        [
            {
                "gender": d["gender"],
                "SeniorCitizen": 1 if d["senior"] == "Yes" else 0,
                "Partner": d["partner"],
                "Dependents": d["dependents"],
                "tenure": d["tenure"],
                "PhoneService": d["phone"],
                "MultipleLines": d["multiple"],
                "InternetService": d["internet"],
                "OnlineSecurity": d["security"],
                "OnlineBackup": d["backup"],
                "DeviceProtection": d["protection"],
                "TechSupport": d["tech"],
                "StreamingTV": d["tv"],
                "StreamingMovies": d["movies"],
                "Contract": d["contract"],
                "PaperlessBilling": d["paperless"],
                "PaymentMethod": d["payment"],
                "MonthlyCharges": d["monthly"],
                "TotalCharges": get_total_charges(),
            }
        ]
    )
    dummies = pd.get_dummies(row)
    return dummies.reindex(columns=columns, fill_value=0).astype(float)

# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
with st.sidebar:
    st.header("Settings")
    threshold = st.slider(
        "Decision threshold",
        min_value=0.05,
        max_value=0.95,
        value=default_threshold,
        step=0.01,
        help="Minimum probability to classify a customer as churn. "
        "Lower = higher recall (catches more churners) and lower precision.",
    )

# ----------------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------------
st.title("Customer Churn Prediction")
st.caption("Answer a few questions about the customer to estimate the risk of cancellation.")
render_stepper(step)
st.write("")

# ----------------------------------------------------------------------------
# Step 1: Profile
# ----------------------------------------------------------------------------
if step == 0:
    with st.container(border=True):
        st.subheader("👤 Profile")
        c1, c2 = st.columns(2)
        with c1:
            select("Gender", "gender", ["Female", "Male"])
            select("Senior citizen", "senior", YES_NO)
        with c2:
            select("Has partner", "partner", YES_NO)
            select("Has dependents", "dependents", YES_NO)
        d["tenure"] = st.slider("Tenure (months with the company)", 0, 72, int(d["tenure"]), key="w_tenure")

    _, right = st.columns([3, 1])
    right.button("Next", on_click=go_to, args=(1,), type="primary", use_container_width=True)

# ----------------------------------------------------------------------------
# Step 2: Services
# ----------------------------------------------------------------------------
elif step == 1:
    with st.container(border=True):
        st.subheader("📞 Services")

        phone = select("Phone service", "phone", YES_NO)
        if phone == "Yes":
            select("Multiple lines", "multiple", YES_NO)
        else:
            d["multiple"] = "No phone service"
            st.caption("Multiple lines: not available without phone service.")

        st.divider()

        internet = select("Internet service", "internet", ["DSL", "Fiber optic", "No"])
        if internet == "No":
            for key, _ in INTERNET_ADDONS:
                d[key] = "No internet service"
            st.caption("Internet add-ons (security, backup, streaming...) are not available without internet service.")
        else:
            c1, c2 = st.columns(2)
            for i, (key, label) in enumerate(INTERNET_ADDONS):
                with (c1 if i % 2 == 0 else c2):
                    select(label, key, YES_NO)

    left, _, right = st.columns([1, 2, 1])
    left.button("Back", on_click=go_to, args=(0,), use_container_width=True)
    right.button("Next", on_click=go_to, args=(2,), type="primary", use_container_width=True)

# ----------------------------------------------------------------------------
# Step 3: Contract and billing
# ----------------------------------------------------------------------------
elif step == 2:
    with st.container(border=True):
        st.subheader("💳 Contract & billing")

        c1, c2 = st.columns(2)
        with c1:
            select("Contract", "contract", ["Month-to-month", "One year", "Two year"])
            select("Paperless billing", "paperless", YES_NO)
        with c2:
            select(
                "Payment method",
                "payment",
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)",
                ],
            )
            d["monthly"] = st.number_input(
                "Monthly charges", 0.0, 200.0, float(d["monthly"]), step=0.5, key="w_monthly"
            )

        d["manual_total"] = st.checkbox(
            "I know the total charges", value=d["manual_total"], key="w_manual_total"
        )
        if d["manual_total"]:
            start = d["total"] if d["total"] is not None else round(d["monthly"] * d["tenure"], 2)
            d["total"] = st.number_input(
                "Total charges", 0.0, 10000.0, float(start), step=1.0, key="w_total"
            )
        else:
            st.caption(
                f"Total charges will be estimated as {get_total_charges():,.2f} "
                "(monthly charges × tenure)."
            )

    left, _, right = st.columns([1, 2, 1])
    left.button("Back", on_click=go_to, args=(1,), use_container_width=True)
    right.button(
        "Predict", on_click=go_to, args=(RESULT_STEP,), type="primary", use_container_width=True
    )

# ----------------------------------------------------------------------------
# Result
# ----------------------------------------------------------------------------
else:
    X_new = build_input()
    proba = float(model.predict_proba(X_new)[:, 1][0])
    is_churn = proba >= threshold

    with st.container(border=True):
        st.subheader("📊 Result")
        c1, c2 = st.columns(2)
        c1.metric("Churn probability", f"{proba:.1%}")
        c2.metric("Threshold used", f"{threshold:.2f}")
        st.progress(min(max(proba, 0.0), 1.0))

        if is_churn:
            st.error("This customer has a **high risk of churn**.")
        else:
            st.success("This customer has a **low risk of churn**.")

    with st.expander("Customer summary"):
        summary = {
            "Gender": d["gender"],
            "Senior citizen": d["senior"],
            "Partner": d["partner"],
            "Dependents": d["dependents"],
            "Tenure (months)": d["tenure"],
            "Phone service": d["phone"],
            "Multiple lines": d["multiple"],
            "Internet service": d["internet"],
            **{label: d[key] for key, label in INTERNET_ADDONS},
            "Contract": d["contract"],
            "Paperless billing": d["paperless"],
            "Payment method": d["payment"],
            "Monthly charges": d["monthly"],
            "Total charges": get_total_charges(),
        }
        st.dataframe(pd.Series(summary, name="value").astype(str))

    with st.expander("View data sent to the model"):
        st.dataframe(X_new.T.rename(columns={0: "value"}))

    left, _, right = st.columns([1, 1, 1])
    left.button("Edit answers", on_click=go_to, args=(0,), use_container_width=True)
    right.button("New prediction", on_click=restart, type="primary", use_container_width=True)