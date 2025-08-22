from flask import Flask, render_template, request, Response
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')   # For non-GUI backends (important in Flask)
import matplotlib.pyplot as plt
import io

app = Flask(__name__)

# Load models
with open("SLR_model.pkl", "rb") as f:
    slr_model = pickle.load(f)

with open("MLR_model.pkl", "rb") as f:
    mlr_model = pickle.load(f)


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    model_type = request.args.get("model")   # 'slr' or 'mlr'
    user_input = {}

    if request.method == "POST":
        try:
            if model_type == "slr":
                years = float(request.form.get("years"))
                prediction = slr_model.predict([[years]])[0]
                prediction = round(float(prediction), 2)
                user_input["years"] = years

            elif model_type == "mlr":
                rnd = float(request.form.get("rnd"))
                admin = float(request.form.get("admin"))
                marketing = float(request.form.get("marketing"))
                state = int(request.form.get("state"))
                features = np.array([[rnd, admin, marketing, state]])
                prediction = mlr_model.predict(features)[0]
                prediction = round(float(prediction), 2)
                user_input = {"rnd": rnd, "admin": admin, "marketing": marketing, "state": state}

        except Exception as e:
            prediction = f"Error: {e}"

    return render_template("index.html", model_type=model_type, prediction=prediction, user_input=user_input)


# ---------- GRAPH ROUTES ----------

@app.route("/plot_slr")
def plot_slr():
    years = float(request.args.get("years", 0))
    salary = slr_model.predict([[years]])[0]

    # Generate line for regression
    x_vals = np.linspace(0, 20, 50).reshape(-1, 1)
    y_vals = slr_model.predict(x_vals)

    plt.figure(figsize=(6,4))
    plt.scatter(years, salary, color="red", label=f"Your Input ({years}, {round(salary,2)})")
    plt.plot(x_vals, y_vals, color="blue", label="Regression Line")
    plt.xlabel("Years of Experience")
    plt.ylabel("Predicted Salary")
    plt.legend()
    plt.tight_layout()

    img = io.BytesIO()
    plt.savefig(img, format="png")
    img.seek(0)
    return Response(img.getvalue(), mimetype="image/png")


@app.route("/plot_mlr")
def plot_mlr():
    rnd = float(request.args.get("rnd", 0))
    admin = float(request.args.get("admin", 0))
    marketing = float(request.args.get("marketing", 0))
    state = int(request.args.get("state", 0))

    features = np.array([[rnd, admin, marketing, state]])
    profit = mlr_model.predict(features)[0]

    # Simple bar chart for visualization
    labels = ["R&D", "Admin", "Marketing", "State Code"]
    values = [rnd, admin, marketing, state]

    plt.figure(figsize=(6,4))
    plt.bar(labels, values, color=["#ff9999","#66b3ff","#99ff99","#ffcc99"])
    plt.title(f"Predicted Profit = {round(profit,2)}")
    plt.ylabel("Values")
    plt.tight_layout()

    img = io.BytesIO()
    plt.savefig(img, format="png")
    img.seek(0)
    return Response(img.getvalue(), mimetype="image/png")


if __name__ == "__main__":
    app.run(debug=True)