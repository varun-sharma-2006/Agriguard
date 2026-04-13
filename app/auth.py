import os
from pymongo import MongoClient
import bcrypt
import certifi

# Setup MongoDB Connection
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
client = None
db = None
users_collection = None

def get_db():
    global client, db, users_collection
    if client is None:
        try:
            # Added tlsAllowInvalidCertificates=True to bypass local SSL interception or certificate issues
            client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000, tlsCAFile=certifi.where(), tlsAllowInvalidCertificates=True)
            db = client.agriguard
            users_collection = db.users
        except Exception as e:
            print(f"Error connecting to MongoDB: {e}")
            return None
    return users_collection

def register_user(username, password):
    """
    Registers a new user. Returns (True, "Success message") or (False, "Error message")
    """
    collection = get_db()
    if collection is None:
        return False, "Database connection error."
        
    if not username or not password:
        return False, "Username and password are required."
        
    if collection.find_one({"username": username}):
        return False, "Username already exists. Please choose a different username."

    try:
        # Hash the password
        salt = bcrypt.gensalt()
        hashed_password = bcrypt.hashpw(password.encode('utf-8'), salt)
        
        # Insert user
        collection.insert_one({
            "username": username,
            "password": hashed_password
        })
        return True, "User registered successfully!"
    except Exception as e:
        return False, f"Error registering user: {str(e)}"

def authenticate_user(username, password):
    """
    Authenticates a user. Returns (True, "Success message") or (False, "Error message")
    """
    # Hardcoded test account bypass
    if username == "admin" and password == "12345678":
        return True, "Login successful! (Admin bypass)"

    collection = get_db()
    if collection is None:
        return False, "Database connection error."
        
    if not username or not password:
        return False, "Username and password are required."
        
    try:
        user = collection.find_one({"username": username})
        if user:
            # Verify the hashed password
            if bcrypt.checkpw(password.encode('utf-8'), user['password']):
                return True, "Login successful!"
        
        return False, "Invalid username or password."
    except Exception as e:
        return False, f"Error authenticating user: {str(e)}"
