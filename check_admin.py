from database.db import db, User
from app import app

with app.app_context():
    admins = User.query.filter_by(is_admin=True).all()
    for admin in admins:
        print(f"Username: {admin.username}, Password: {admin.password}")