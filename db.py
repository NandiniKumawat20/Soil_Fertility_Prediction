import os
from datetime import datetime
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, DuplicateKeyError
import bcrypt


# ---- MongoDB Connection ----
MONGO_URI = os.environ.get('MONGO_URI', 'mongodb://localhost:27017/')
DB_NAME = 'soil_fertility_db'

try:
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
    client.admin.command('ping')
    db = client[DB_NAME]
    users_collection = db['users']
    users_collection.create_index('email', unique=True)
    activity_collection = db['activity_log']
    activity_collection.create_index('email')
    activity_collection.create_index('timestamp')
    MONGO_CONNECTED = True
    print("[DB] Connected to MongoDB successfully")
except Exception as e:
    client = None
    db = None
    users_collection = None
    activity_collection = None
    MONGO_CONNECTED = False
    print(f"[DB] MongoDB connection failed: {e}")


def hash_password(password):
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())


def check_password(password, hashed):
    return bcrypt.checkpw(password.encode('utf-8'), hashed)


def create_user(first_name, last_name, email, password, occupation, country, phone):
    if not MONGO_CONNECTED:
        return None, "Database is not connected"

    if users_collection.find_one({'email': email}):
        return None, "Email already registered"

    user_doc = {
        'firstName': first_name,
        'lastName': last_name,
        'email': email,
        'password': hash_password(password),
        'occupation': occupation,
        'country': country,
        'phone': phone,
        'createdAt': datetime.utcnow(),
        'updatedAt': datetime.utcnow()
    }

    try:
        users_collection.insert_one(user_doc)
    except DuplicateKeyError:
        return None, "Email already registered"

    return {
        'firstName': first_name,
        'lastName': last_name,
        'email': email,
        'occupation': occupation,
        'country': country,
        'phone': phone
    }, None


def authenticate_user(email, password):
    if not MONGO_CONNECTED:
        return None, "Database is not connected"

    user = users_collection.find_one({'email': email})
    if not user:
        return None, "Invalid email or password"

    if not check_password(password, user['password']):
        return None, "Invalid email or password"

    return {
        'firstName': user['firstName'],
        'lastName': user['lastName'],
        'email': user['email'],
        'occupation': user.get('occupation', ''),
        'country': user.get('country', ''),
        'phone': user.get('phone', '')
    }, None


def get_user_by_email(email):
    if not MONGO_CONNECTED:
        return None, "Database is not connected"

    user = users_collection.find_one({'email': email})
    if not user:
        return None, "User not found"

    return {
        'firstName': user['firstName'],
        'lastName': user['lastName'],
        'email': user['email'],
        'occupation': user.get('occupation', ''),
        'country': user.get('country', ''),
        'phone': user.get('phone', ''),
        'createdAt': user.get('createdAt', '').isoformat() if user.get('createdAt') else ''
    }, None


def update_user(email, update_fields):
    if not MONGO_CONNECTED:
        return None, "Database is not connected"

    allowed = ['firstName', 'lastName', 'occupation', 'country', 'phone']
    fields_to_set = {k: v.strip() for k, v in update_fields.items() if k in allowed and v}

    if not fields_to_set:
        return None, "No valid fields to update"

    fields_to_set['updatedAt'] = datetime.utcnow()

    result = users_collection.update_one(
        {'email': email},
        {'$set': fields_to_set}
    )

    if result.matched_count == 0:
        return None, "User not found"

    return get_user_by_email(email)


def log_activity(email, action, details=None):
    if not MONGO_CONNECTED:
        return

    activity_doc = {
        'email': email,
        'action': action,
        'details': details or {},
        'timestamp': datetime.utcnow()
    }

    try:
        activity_collection.insert_one(activity_doc)
    except Exception:
        pass


def get_user_activity(email, limit=50):
    if not MONGO_CONNECTED:
        return [], "Database is not connected"

    try:
        cursor = activity_collection.find(
            {'email': email},
            {'_id': 0}
        ).sort('timestamp', -1).limit(limit)

        activities = []
        for doc in cursor:
            doc['timestamp'] = doc['timestamp'].isoformat()
            activities.append(doc)

        return activities, None
    except Exception as e:
        return [], str(e)


def get_user_stats(email):
    if not MONGO_CONNECTED:
        return None, "Database is not connected"

    try:
        pipeline = [
            {'$match': {'email': email}},
            {'$group': {'_id': '$action', 'count': {'$sum': 1}}}
        ]
        results = list(activity_collection.aggregate(pipeline))

        stats = {
            'soil_analysis': 0,
            'soil_reports': 0,
            'fertilizer_predictions': 0,
            'fertilizer_selector': 0,
            'chatbot': 0,
            'total_actions': 0
        }

        for r in results:
            action = r['_id']
            count = r['count']
            if action == 'soil_prediction':
                stats['soil_analysis'] = count
            elif action == 'soil_report':
                stats['soil_reports'] = count
            elif action == 'fertilizer_prediction':
                stats['fertilizer_predictions'] = count
            elif action == 'fertilizer_selector':
                stats['fertilizer_selector'] = count
            elif action == 'chat':
                stats['chatbot'] = count
            stats['total_actions'] += count

        return stats, None
    except Exception as e:
        return None, str(e)


def get_user_monthly_activity(email):
    if not MONGO_CONNECTED:
        return None, "Database is not connected"

    try:
        from datetime import timedelta
        now = datetime.utcnow()
        six_months_ago = now - timedelta(days=180)

        pipeline = [
            {'$match': {'email': email, 'timestamp': {'$gte': six_months_ago}}},
            {'$group': {
                '_id': {
                    'year': {'$year': '$timestamp'},
                    'month': {'$month': '$timestamp'},
                    'action': '$action'
                },
                'count': {'$sum': 1}
            }},
            {'$sort': {'_id.year': 1, '_id.month': 1}}
        ]
        results = list(activity_collection.aggregate(pipeline))

        months = []
        soil_data = []
        fert_data = []
        chat_data = []

        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

        for i in range(5, -1, -1):
            d = now - timedelta(days=30 * i)
            label = month_names[d.month - 1] + ' ' + str(d.year)[2:]
            months.append(label)
            soil_data.append(0)
            fert_data.append(0)
            chat_data.append(0)

        for r in results:
            y = r['_id']['year']
            m = r['_id']['month']
            action = r['_id']['action']
            count = r['count']

            for i in range(5, -1, -1):
                d = now - timedelta(days=30 * i)
                if d.year == y and d.month == m:
                    idx = 5 - i
                    if action in ('soil_prediction', 'soil_report'):
                        soil_data[idx] += count
                    elif action in ('fertilizer_prediction', 'fertilizer_selector'):
                        fert_data[idx] += count
                    elif action == 'chat':
                        chat_data[idx] += count
                    break

        return {
            'labels': months,
            'soil_analysis': soil_data,
            'fertilizer': fert_data,
            'chatbot': chat_data
        }, None
    except Exception as e:
        return None, str(e)


def get_recent_soil_analyses(email, limit=5):
    if not MONGO_CONNECTED:
        return [], "Database is not connected"

    try:
        cursor = activity_collection.find(
            {
                'email': email,
                'action': 'soil_prediction',
                'details.inputs': {'$exists': True}
            },
            {'_id': 0, 'details': 1, 'timestamp': 1}
        ).sort('timestamp', -1).limit(limit)

        analyses = []
        for doc in cursor:
            entry = doc.get('details', {})
            ts = doc.get('timestamp', '')
            if hasattr(ts, 'isoformat'):
                ts = ts.isoformat()
            entry['timestamp'] = ts
            analyses.append(entry)

        return analyses, None
    except Exception as e:
        return [], str(e)
