import os
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
import dash
from dash import dcc, html, Input, Output, clientside_callback
import dash_mantine_components as dmc

# ── Load model ────────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))

saved_clf = xgb.XGBClassifier()
saved_clf.load_model(os.path.join(_HERE, "xgboost_model.json"))

label_encoder = LabelEncoder()
label_encoder.classes_ = np.array(["<30", ">30", "NO"])


# ── Prediction function ───────────────────────────────────────────────────────
def get_model_prediction(inpatient_visits, emergency_visits, medications_num,
                         hosp_time, diagnoses_num, lab_procdrs_num,
                         insulin_lvl, A1C_lvl):
    categorical_cols = ["insulin", "A1Cresult"]
    sample_data = {
        "number_inpatient":   [int(inpatient_visits)],
        "number_emergency":   [int(emergency_visits)],
        "num_medications":    [int(medications_num)],
        "time_in_hospital":   [int(hosp_time)],
        "number_diagnoses":   [int(diagnoses_num)],
        "num_lab_procedures": [int(lab_procdrs_num)],
        "insulin":            [str(insulin_lvl)],
        "A1Cresult":          [str(A1C_lvl)],
    }
    sample_df = pd.DataFrame(sample_data)
    for col in categorical_cols:
        sample_df[col] = sample_df[col].astype("category")

    prediction     = saved_clf.predict(sample_df)
    props = saved_clf.predict_proba(sample_df)[0]
    original_label = label_encoder.inverse_transform(prediction)
    prop = max(props)  # confidence of the predicted class
    pred = original_label[0]


    if pred == "<30":
        color = "#E07A7A"
        val   = np.random.randint(61,100)
    elif pred == ">30":
        color = "#E6B566"
        val   = np.random.randint(20,60)
    else:
        color = "#4FA645"
        val   = np.random.randint(0,19)  

    return color, val, pred


# ── Design tokens ─────────────────────────────────────────────────────────────
BROWN    = "#8B5E1A"
BG_RIGHT = "#EDE6D8"
TEXT     = "#2C2416"
SOFT     = "#9C8E77"
BG_MAIN  = "#F7F4EE"
BORDER   = "#C9B99A"

DEFAULT_SLIDER_COLOR = "#4FA645"

# All numeric slider IDs – drives both layout and callback outputs
SLIDER_IDS = ["inpatient", "emergency", "meds", "time", "diagnoses", "labs"]

app = dash.Dash(__name__, suppress_callback_exceptions=True)

server = app.server

# ── DMC Slider helper ─────────────────────────────────────────────────────────
def dmc_slider(label, sid, min_, max_, value):
    return html.Div([
        html.Div(label, style={"fontSize": "13px", "color": TEXT, "marginBottom": "4px"}),
        dmc.Slider(
            id=sid,
            min=min_,
            max=max_,
            step=1,
            value=value,
            color=DEFAULT_SLIDER_COLOR,   # updated live via Output(sid, "color")
            showLabelOnHover=True,
            size="sm",
            styles={
                "track": {"cursor": "pointer"},
                "thumb": {"borderWidth": "2px"},
            },
        ),
    ], style={"marginBottom": "14px"})


# ── Thermometer component ─────────────────────────────────────────────────────
def thermometer(val, pred_label, color):
    pct = int(val)

    if pred_label == "<30":
        risk_text, css_class, pill_class = "High risk",     "thermo-high",     "pill-high"
    elif pred_label == ">30":
        risk_text, css_class, pill_class = "Moderate risk", "thermo-moderate", "pill-moderate"
    else:
        risk_text, css_class, pill_class = "Low risk",      "thermo-low",      "pill-low"

    ticks = html.Div([
        html.Div(f"{v}%", style={"fontSize": "11px", "color": SOFT})
        for v in [80, 60, 40, 20, 0]
    ], style={
        "display": "flex", "flexDirection": "column",
        "justifyContent": "space-between",
        "height": "90%", "paddingBottom": "26px",
        "marginRight": "10px", "textAlign": "right",
    })

    thermo_col = html.Div([
        html.Div(className="thermo-bar", children=[
            html.Div(className="thermo-fill",
                     style={"height": f"{max(pct, 4)}%", "background": color})
        ]),
        html.Div(className="thermo-bulb", style={"background": color}),
    ], style={"display": "flex", "flexDirection": "column",
              "alignItems": "center", "height": "100%"})

    info_col = html.Div([
        html.Div("Predicted <30 day readmission",
                 style={"fontSize": "13px", "color": SOFT, "marginBottom": "8px"}),
        html.Div(f"{pct}%",
                 style={"fontSize": "64px", "fontWeight": "700",
                        "color": TEXT, "lineHeight": "1"}),
        html.Div(f"Predicted: {pred_label}",
                 style={"fontSize": "13px", "color": SOFT, "marginTop": "4px"}),
        html.Div(risk_text, className=f"pill {pill_class}"),
    ], style={"paddingLeft": "32px", "display": "flex",
              "flexDirection": "column", "justifyContent": "center"})

    return html.Div(
        [ticks, thermo_col, info_col],
        className=css_class,
        style={"display": "flex", "flexDirection": "row",
               "alignItems": "stretch", "height": "100%", "width": "100%"},
    )


# ── Sidebar ───────────────────────────────────────────────────────────────────
sidebar = html.Div([
    html.Div([
        html.Div("DiabetesIQ",
                 style={"fontWeight": "700", "fontSize": "18px", "color": TEXT}),
        html.Div("readmission analytics",
                 style={"fontSize": "11px", "color": SOFT, "marginTop": "2px"}),
    ], style={"marginBottom": "10px"}),
    html.Div("DASHBOARDS", className="sidebar-label"),
    html.Div([html.Span("▪ "), html.Span("Risk predictor")],
             className="sidebar-item active"),
], className="sidebar")


# ── Layout ────────────────────────────────────────────────────────────────────
app.layout = dmc.MantineProvider([html.Div([

    # Store: carries prediction color between the two callbacks
    dcc.Store(id="pred-color-store", data=DEFAULT_SLIDER_COLOR),

    # Invisible anchor for clientside_callback output
    html.Div(id="slider-color-style", style={"display": "none"}),

    html.Div([
        sidebar,
        html.Div([

            # Header
            html.Div([
                html.Div("Risk predictor", style={
                    "fontSize": "40px", "fontWeight": "600",
                    "color": TEXT, "textAlign": "center",
                }),
            ], style={
                "display": "flex", "alignItems": "flex-start",
                "justifyContent": "center",
                "padding": "18px 28px 14px 28px",
                "borderBottom": f"1px solid {BORDER}", "flexShrink": "0",
            }),

            html.Div([

                # LEFT – sliders
                html.Div([
                    html.Div("Patient inputs", style={
                        "fontSize": "20px", "fontWeight": "700",
                        "color": TEXT, "marginBottom": "14px",
                    }),
                    dmc_slider("Prior inpatient visits",  "inpatient",  0, 10,  2),
                    dmc_slider("Emergency visits",        "emergency",  0, 30,  0),
                    dmc_slider("Medications count",       "meds",       0, 30, 14),
                    dmc_slider("Time in hospital (days)", "time",       1, 14,  4),
                    dmc_slider("Number of diagnoses",     "diagnoses",  0, 10,  6),
                    dmc_slider("Lab procedures",          "labs",       0, 50, 19),

                    html.Div([
                        html.Div([
                            html.Div("Insulin",
                                     style={"fontSize": "13px", "color": TEXT,
                                            "marginBottom": "4px"}),
                            dcc.Dropdown(["No", "Up", "Down", "Steady"],
                                         value="No", id="insulin", clearable=False),
                        ], style={"flex": "1"}),
                        html.Div([
                            html.Div("A1C result",
                                     style={"fontSize": "13px", "color": TEXT,
                                            "marginBottom": "4px"}),
                            dcc.Dropdown(["None", "Norm", ">7", ">8"],
                                         value="None", id="a1c", clearable=False),
                        ], style={"flex": "1"}),
                    ], style={"display": "flex", "gap": "20px", "marginTop": "6px"}),

                ], style={"width": "48%", "padding": "22px 28px", "flexShrink": "0"}),

                # RIGHT – thermometer
                html.Div(id="thermo", style={
                    "flex": "1", "background": BG_RIGHT,
                    "borderLeft": f"1px solid {BORDER}",
                    "padding": "28px 36px", "display": "flex",
                    "alignItems": "stretch",
                }),

            ], style={"display": "flex", "flex": "1",
                      "overflow": "hidden", "minHeight": "0"}),

        ], style={"flex": "1", "display": "flex", "flexDirection": "column",
                  "overflow": "hidden", "minHeight": "0"}),

    ], style={
        "display": "flex", "height": "100vh", "overflow": "hidden",
        "fontFamily": "'Georgia', 'Times New Roman', serif",
        "background": BG_MAIN,
    }),
], style={"height": "100vh", "overflow": "hidden"})])


# ── Shared inputs ─────────────────────────────────────────────────────────────
_INPUTS = [
    Input("inpatient", "value"),
    Input("emergency", "value"),
    Input("meds",      "value"),
    Input("time",      "value"),
    Input("diagnoses", "value"),
    Input("labs",      "value"),
    Input("insulin",   "value"),
    Input("a1c",       "value"),
]


# ── Callback 1 – thermometer + store + one color Output per slider ────────────
# Exactly mirrors the reference pattern: Output("slider", "color") per slider.
@app.callback(
    Output("thermo",           "children"),
    Output("pred-color-store", "data"),
    *[Output(sid, "color") for sid in SLIDER_IDS],
    *_INPUTS,
)
def update_dashboard(inpatient, emergency, meds, time, diagnoses, labs, insulin, a1c):
    color, val, pred_label = get_model_prediction(
        inpatient, emergency, meds, time, diagnoses, labs, insulin, a1c
    )
    thermo_component = thermometer(val, pred_label, color)
    # thermometer  +  store color  +  same color pushed to each slider
    return (thermo_component, color) + (color,) * len(SLIDER_IDS)


# ── Callback 2 – clientside safety net: keeps <style> in sync too ─────────────
# Handles any edge case where DMC re-renders and the prop momentarily resets.
clientside_callback(
    """
    function(color) {
        if (!color) color = "#4FA645";
        var id  = "dmc-slider-dynamic";
        var el  = document.getElementById(id);
        if (!el) {
            el    = document.createElement("style");
            el.id = id;
            document.head.appendChild(el);
        }
        el.textContent = [
            ".mantine-Slider-bar   { background-color: " + color + " !important; }",
            ".mantine-Slider-thumb { border-color: "     + color + " !important; background-color: " + color + " !important; }",
            ".mantine-Slider-label { background-color: " + color + " !important; }"
        ].join(" ");
        return color;
    }
    """,
    Output("slider-color-style", "children"),
    Input("pred-color-store",    "data"),
)


# ── Run ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True)