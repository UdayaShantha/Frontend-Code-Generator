import json
import pymongo
from pymongo import MongoClient
import os
import sys
import urllib.parse


def upload_to_mongodb(json_file, connection_string=None, db_name="web_design_ml", collection_name="website_data",
                      auth_db="admin"):
    """
    Upload the scraped website dataset to MongoDB

    Parameters:
    - json_file: Path to the JSON file containing the website data
    - connection_string: MongoDB connection string
    - db_name: Name of the database
    - collection_name: Name of the collection

    If connection_string is not provided, it will look for MONGO_URI environment variable
    """
    try:
        # Get MongoDB connection string
        if connection_string is None:
            connection_string = os.environ.get("MONGO_URI")
            if connection_string is None:
                print(
                    "Error: No MongoDB connection string provided. Please set MONGO_URI environment variable or provide as an argument.")
                sys.exit(1)

        # Connect to MongoDB
        print(f"Connecting to MongoDB...")
        print(
            f"Using connection string (with password hidden): {connection_string.replace(connection_string.split('@')[0].split(':')[-1], '******') if '@' in connection_string else connection_string}")

        # Try to parse the connection string to extract auth_db if present
        try:
            uri_parts = pymongo.uri_parser.parse_uri(connection_string)
            if 'authSource' in uri_parts['options']:
                auth_db = uri_parts['options']['authSource']
                print(f"Using authentication database: {auth_db}")
        except Exception as e:
            print(f"Warning: Could not parse connection string: {e}")

        # Try to connect with explicit auth DB parameter
        try:
            client = MongoClient(connection_string)
            # Test connection
            client.admin.command('ping')
            print("MongoDB connection successful!")
            db = client[db_name]
            collection = db[collection_name]
        except pymongo.errors.OperationFailure as e:
            print(f"MongoDB authentication error: {e}")
            print("Trying alternative connection methods...")

            # If we have a username/password in the connection string but no authSource
            if '@' in connection_string and '?' not in connection_string:
                # Add authSource parameter
                new_conn_str = f"{connection_string}?authSource={auth_db}"
                print(
                    f"Trying with explicit authSource: {new_conn_str.replace(new_conn_str.split('@')[0].split(':')[-1], '******')}")
                client = MongoClient(new_conn_str)
                db = client[db_name]
                collection = db[collection_name]

        # Load the JSON file
        print(f"Loading data from {json_file}...")
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Format the data for MongoDB
        formatted_data = []
        for item in data:
            # Make sure Object_Id is properly formatted for MongoDB
            if "Object_Id" in item:
                item["_id"] = item["Object_Id"]
                del item["Object_Id"]

            formatted_data.append(item)

        # Insert data in batches to avoid memory issues
        batch_size = 50
        total_count = len(formatted_data)

        print(f"Uploading {total_count} documents to MongoDB in batches of {batch_size}...")

        for i in range(0, total_count, batch_size):
            batch = formatted_data[i:i + batch_size]

            # Handle potential duplicate _id errors
            for doc in batch:
                try:
                    collection.replace_one({"_id": doc["_id"]}, doc, upsert=True)
                except pymongo.errors.DuplicateKeyError:
                    print(f"Warning: Duplicate key found for {doc.get('Name', 'unknown')}. Updating existing document.")
                    # Generate a new ID
                    import hashlib
                    import time
                    new_id = hashlib.md5(f"{doc['Name']}_{time.time()}".encode()).hexdigest()[:24]
                    doc["_id"] = new_id
                    collection.insert_one(doc)

            print(f"Uploaded batch {i // batch_size + 1}/{(total_count - 1) // batch_size + 1}")

        # Create indexes for faster queries
        print("Creating indexes...")
        collection.create_index("Name")
        collection.create_index("Type")

        print(f"Successfully uploaded dataset to MongoDB: {db_name}.{collection_name}")
        print(f"Total documents: {collection.count_documents({})}")

    except Exception as e:
        print(f"Error uploading to MongoDB: {e}")
        sys.exit(1)


def main():
    # Check command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Upload website dataset to MongoDB')
    parser.add_argument('json_file', help='Path to the JSON file containing website data')
    parser.add_argument('--connection', '-c', help='MongoDB connection string')
    parser.add_argument('--database', '-d', default='web_design_ml', help='MongoDB database name')
    parser.add_argument('--collection', '-n', default='website_data', help='MongoDB collection name')
    parser.add_argument('--auth-db', '-a', default='admin', help='Authentication database name')
    parser.add_argument('--username', '-u', help='MongoDB username')
    parser.add_argument('--password', '-p', help='MongoDB password')
    parser.add_argument('--host', '-h', default='localhost', help='MongoDB host')
    parser.add_argument('--port', default=27017, type=int, help='MongoDB port')

    args = parser.parse_args()

    # Build connection string if individual components are provided
    connection_string = args.connection
    if not connection_string and args.username and args.password:
        escaped_username = urllib.parse.quote_plus(args.username)
        escaped_password = urllib.parse.quote_plus(args.password)
        connection_string = f"mongodb://{escaped_username}:{escaped_password}@{args.host}:{args.port}/{args.database}?authSource={args.auth_db}"

    # Upload the data
    upload_to_mongodb(
        args.json_file,
        connection_string=connection_string,
        db_name=args.database,
        collection_name=args.collection,
        auth_db=args.auth_db
    )


if __name__ == "__main__":
    main()