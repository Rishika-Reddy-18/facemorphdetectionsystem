from flask import Flask, render_template, request, redirect, session
from database.db import db, User, History
import tensorflow as tf
import numpy as np
import cv2
import os

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///db.sqlite3'
app.config['SECRET_KEY'] = "supersecretkey123"

db.init_app(app)

# ================== LOAD MODEL ==================
model = tf.keras.models.load_model("models/morph_model.h5")
dummy = np.zeros((1,128,128,3), dtype=np.float32)
_ = model(dummy, training=False)

# ================== HEATMAP FUNCTION ==================
def get_heatmap(model, img_array):
    if len(img_array.shape) == 3:
        img_array = np.expand_dims(img_array, axis=0)

    _ = model(img_array, training=False)

    last_conv_idx = None
    for i, layer in reversed(list(enumerate(model.layers))):
        if isinstance(layer, tf.keras.layers.Conv2D):
            last_conv_idx = i
            break
    if last_conv_idx is None:
        return None

    inputs = tf.keras.Input(shape=img_array.shape[1:])
    x = inputs
    for layer in model.layers[:last_conv_idx+1]:
        x = layer(x)

    conv_output = x
    y = x
    for layer in model.layers[last_conv_idx+1:]:
        y = layer(y)

    grad_model = tf.keras.models.Model(inputs=inputs, outputs=[conv_output, y])

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, 0]

    grads = tape.gradient(loss, conv_outputs)
    if grads is None:
        return None

    pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))
    conv_outputs = conv_outputs[0]

    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap).numpy()

    heatmap = np.maximum(heatmap, 0)
    if np.max(heatmap) != 0:
        heatmap /= np.max(heatmap)

    return heatmap

# ================== LOGIN ==================
@app.route("/", methods=["GET","POST"])
def login():
    if request.method == "POST":
        user = User.query.filter_by(username=request.form['username']).first()

        if user and user.password == request.form['password']:
            session['user'] = user.id
            session['admin'] = user.is_admin  # ✅ use DB value

            # Redirect based on role
            if user.is_admin:
                return redirect("/admin")
            else:
                return redirect("/dashboard")
        else:
            return "Invalid username or password"

    return render_template("login.html")

# ================== REGISTER ==================
@app.route("/register", methods=["GET","POST"])
def register():
    if request.method=="POST":
        existing = User.query.filter_by(username=request.form['username']).first()
        if existing:
            return "Username already exists!"

        is_admin = True if request.form['username'].lower() == "admin" else False

        new_user = User(
            username=request.form['username'],
            password=request.form['password'],
            is_admin=is_admin
        )

        db.session.add(new_user)
        db.session.commit()

        return redirect("/")

    return render_template("register.html")

# ================== DASHBOARD ==================
@app.route("/dashboard", methods=["GET","POST"])
def dashboard():
    if not session.get('user'):
        return redirect("/")

    heatmap_path = None
    result = None
    confidence = None
    image_path = None

    if request.method=="POST":
        file = request.files['image']
        upload_folder = "static/uploads"
        os.makedirs(upload_folder, exist_ok=True)
        path = os.path.join(upload_folder, file.filename)
        file.save(path)

        img = cv2.imread(path)
        img = cv2.resize(img,(128,128))
        img_array = img.astype(np.float32)/255.0
        img_array_exp = np.expand_dims(img_array, axis=0)

        pred = float(model.predict(img_array_exp)[0][0])

        if pred >= 0.5:
            result = "Morphed"
            confidence = round(pred*100,2)
        else:
            result = "Genuine"
            confidence = round((1-pred)*100,2)

        if result=="Morphed":
            heatmap = get_heatmap(model,img_array_exp)
            if heatmap is not None:
                heatmap = cv2.resize(heatmap,(128,128))
                heatmap = np.uint8(255*heatmap)
                heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                original = cv2.imread(path)
                original = cv2.resize(original,(128,128))
                superimposed = cv2.addWeighted(original,0.6,heatmap,0.4,0)
                heatmap_path = os.path.join(upload_folder,"heatmap_"+file.filename)
                cv2.imwrite(heatmap_path, superimposed)

        history = History(
            user_id=session['user'],
            image_path=path,
            result=result,
            confidence=confidence
        )
        db.session.add(history)
        db.session.commit()
        image_path = path

    return render_template("dashboard.html",
                           result=result,
                           confidence=confidence,
                           image=image_path,
                           heatmap=heatmap_path)

# ================== ADMIN ==================
@app.route("/admin")
def admin():
    if not session.get('admin', False):
        return "Access Denied: Admins only"

    total_detections = History.query.count()
    morphed_detections = History.query.filter_by(result="Morphed").count()
    morphed_confidences = [h.confidence for h in History.query.filter_by(result="Morphed").all()]
    detection_accuracy = round(sum(morphed_confidences)/len(morphed_confidences), 2) if morphed_confidences else 0
    total_users = User.query.count()
    history_data = History.query.order_by(History.created_at.desc()).all()

    return render_template(
        "admin.html",
        total_detections=total_detections,
        morphed_detections=morphed_detections,
        detection_accuracy=detection_accuracy,
        total_users=total_users,
        history_data=history_data
    )

# ================== CHATBOT ==================
@app.route("/chat", methods=["POST"])
def chat():
    msg = request.form.get("msg","").lower()

    if "morph" in msg:
        reply="A morphed face is a digitally altered image."
    elif "genuine" in msg:
        reply="A genuine face is an original image."
    elif "heatmap" in msg:
        reply="Heatmaps show where AI is focusing."
    else:
        reply="Ask about morph, genuine, or heatmap."

    return reply

# ================== LOGOUT ==================
@app.route("/logout")
def logout():
    session.clear()
    return redirect("/")

# ================== RUN ==================
if __name__=="__main__":
    with app.app_context():
        db.create_all()

        # ✅ Create default admin if missing
        admin = User.query.filter_by(username="admin").first()
        if not admin:
            admin_user = User(username="admin", password="admin123", is_admin=True)
            db.session.add(admin_user)
            db.session.commit()
            print("Default admin created: username='admin', password='admin123'")

    app.run(debug=True)