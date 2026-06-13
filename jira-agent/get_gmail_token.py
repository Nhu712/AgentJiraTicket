"""
Run this script ONCE locally to get Gmail OAuth2 tokens.
Then copy the printed tokens into your .env file.

Usage:
  python get_gmail_token.py --client-id YOUR_ID --client-secret YOUR_SECRET
"""
import argparse
from google_auth_oauthlib.flow import InstalledAppFlow

SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]

parser = argparse.ArgumentParser()
parser.add_argument("--client-id", required=True)
parser.add_argument("--client-secret", required=True)
args = parser.parse_args()

client_config = {
    "installed": {
        "client_id": args.client_id,
        "client_secret": args.client_secret,
        "auth_uri": "https://accounts.google.com/o/oauth2/auth",
        "token_uri": "https://oauth2.googleapis.com/token",
        "redirect_uris": ["urn:ietf:wg:oauth:2.0:oob", "http://localhost"],
    }
}

flow = InstalledAppFlow.from_client_config(client_config, SCOPES)
creds = flow.run_local_server(port=0)

print("\n✅ Copy các giá trị sau vào file .env:\n")
print(f"GMAIL_ACCESS_TOKEN={creds.token}")
print(f"GMAIL_REFRESH_TOKEN={creds.refresh_token}")
