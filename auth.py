from pydrive2.auth import GoogleAuth

gauth = GoogleAuth()

# Safely initialize the oauth_config dictionary if it doesn't exist
if 'oauth_config' not in gauth.settings:
    gauth.settings['oauth_config'] = {}

# Force offline access and consent prompt to get the refresh token
gauth.settings['oauth_config']['access_type'] = 'offline'
gauth.settings['oauth_config']['prompt'] = 'consent'

gauth.LoadClientConfigFile(r"C:\Users\Amit\Desktop\drone-agent\client_secrets.json")
gauth.CommandLineAuth()
gauth.SaveCredentialsFile(r"C:\Users\Amit\Desktop\drone-agent\saved_creds.json")
print("\n[OK] Authentication successful! New token saved with refresh_token.")