"""Application entry point — initialises and runs the Dash server."""
import dash
import dash_bootstrap_components as dbc

from callbacks import env
from layout import create_layout
from callbacks import register_callbacks

# Dash application setup
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    title="5G Handover Dashboard",
    update_title=None,
    suppress_callback_exceptions=True,
)

app.layout = create_layout(env)
register_callbacks(app)

if __name__ == "__main__":
    app.run(debug=False, host="127.0.0.1", port=8050)
