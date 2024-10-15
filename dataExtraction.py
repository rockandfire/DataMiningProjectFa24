import mysql.connector
import pandas as pd

def connect_to_database():
    try:
        connection = mysql.connector.connect(
            host='localhost',
            database='magic_cards',
            user='root',
            password='thisWasMine'
        )
        if connection.is_connected():
            print("Connected to MySQL database")
            return connection
    except mysql.connector.Error as e:
        print(f"Error while connecting to MySQL: {e}")
    return None

def export_to_csv(connection, filename='magic_cards.csv'):
    query = """
    SELECT name, colorIdentity, keywords, supertypes
    FROM cards
    """
    df = pd.read_sql(query, connection)
    df.to_csv(filename, index=False)
    print(f"Data exported to {filename}")

# Main execution
connection = connect_to_database()
if connection:
    export_to_csv(connection)
    connection.close()
else:
    print("Failed to connect to the database.")