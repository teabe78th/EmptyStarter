
from app import app, db

def init_db():
    with app.app_context():
        # Import all models
        from models import User, Conversation, PsychologicalAnalysis, ReminderLog
        
        # Create all tables
        db.drop_all()  # First drop all existing tables
        db.create_all()  # Create new tables with updated schema
        
        print("Database initialized successfully!")

if __name__ == "__main__":
    init_db()
