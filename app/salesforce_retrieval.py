from configparser import ConfigParser
from simple_salesforce import Salesforce

# Read the secret keys from the configuration file
config = ConfigParser()
config.read("config.ini")
USERNAME = config.get('credentials', 'USERNAME')
PASSWORD = config.get('credentials', 'PASSWORD')
TOKEN = config.get('credentials', 'TOKEN')

"""
Retrieves data from Salesforce and prints information 
about cases and related email messages.
"""
def retrieve_data():
    """

    Returns:
    - generator: A generator object yielding records from the 
    Salesforce query.
    """
    sf = Salesforce(
        username=USERNAME,
        password=PASSWORD,
        security_token=TOKEN
        )

    # Retrieve Data from Salesforce 
    data = sf.query_all_iter("""
    Select Id, Subject, Status, Type, Description,
            (SELECT Id FROM CaseComments),
            (SELECT Subject, TextBody FROM EmailMessages)
    From Case
    Where Status = 'Waiting on Customer'
    """)

    for row in data:
        print("Id:", row['Id'])
        print("Status:", row['Status'])
        print("Type:", row['Type'])
        print("Subject:", row['Subject'])
        print("Description:", row['Description'])
        print("Comments:", row['CaseComments'])
        if row['EmailMessages'] is not None:
            for email in row['EmailMessages']['records']:
                print("Email:")
                print("\tSubject:", email['Subject'])
                print("\tBody:", email['TextBody'])
        else:
            print(f"Case {row['Id']} has no emails.")
        print('\n')
    return data