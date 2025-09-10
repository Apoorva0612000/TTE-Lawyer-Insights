import os

import pandas as pd
import psycopg2
from dotenv import load_dotenv
from sshtunnel import SSHTunnelForwarder

load_dotenv()

with SSHTunnelForwarder(
    (os.getenv("SSH_HOST"), int(os.getenv("SSH_PORT"))),
    ssh_username=os.getenv("SSH_USER"),
    ssh_password=os.getenv("SSH_PASSWORD"),  # or use password
    remote_bind_address=(os.getenv("JUMPBOX_IP"), int(os.getenv("PORT"))),
    local_bind_address=(os.getenv("DB_HOST"), int(os.getenv("DB_PORT"))),
) as tunnel:
    # Connect to your PostgreSQL database
    conn = psycopg2.connect(
        dbname=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        host=os.getenv("DB_HOST"),  # e.g., "localhost" or AWS endpoint
        port=os.getenv("DB_PORT"),  # default PostgreSQL port
    )

    # reading the data
    cur = conn.cursor()
    # Create a cursor to execute SQL
    print("Connection to remote PostgreSQL database successful!")
    # code from here   and
    query = "SELECT tickets.id AS ticket_id, tickets.created_at + interval '5 hours 30 minutes' AS ticket_created_at, services.id AS service_id, Meeting_Compositions.transcription  as transcription, CASE WHEN ps.ticket_id IS NOT NULL THEN TRUE ELSE FALSE END AS proposal_sent, \
    CASE WHEN pa.ticket_id IS NOT NULL THEN TRUE ELSE FALSE END AS proposal_accepted \
    FROM tickets\
    LEFT JOIN city_services ON city_services.id = tickets.city_service_id \
    LEFT JOIN services ON services.id = city_services.service_id \
    LEFT JOIN event_and_reminders ON event_and_reminders.ticket_id  =  tickets.id  \
    LEFT JOIN Meeting_Associations ON Meeting_Associations.associable_id =  event_and_reminders.id \
    LEFT JOIN Meeting_Compositions ON  Meeting_Compositions.virtual_meeting_id =   Meeting_Associations.virtual_meeting_id \
    LEFT JOIN ( \
        SELECT DISTINCT ticket_id \
        FROM customer_case_details \
        WHERE case_details ->> 'libra_proposal_sent' = 'true' \
    ) ps \
        ON ps.ticket_id = tickets.id \
    LEFT JOIN ( \
        SELECT ticket_id \
        FROM activity_events \
        WHERE activity_type = 'proposal_approved_by_customer' \
    ) pa \
        ON pa.ticket_id = tickets.id \
    WHERE services.id IN (738 ,161) AND Meeting_Compositions.transcription IS NOT NULL \
    AND tickets.created_at >= '2025-04-01';"

    df = pd.read_sql(query, conn)

    print(df.head())
    print(df.shape)

    # df.to_csv("/Users/mac/Desktop/VakilSearch/TTE-Lawyer Insights/Data/transcripts.csv", index=False)

    # # Get table names
    # cur.execute("""
    #     SELECT table_name
    #     FROM information_schema.tables
    #     WHERE table_schema = 'public';
    # """)

    # tables = cur.fetchall()
    # print("Tables in the database:")
    # for table in tables:
    #     print(table[0])

    cur.close()
    conn.close()
