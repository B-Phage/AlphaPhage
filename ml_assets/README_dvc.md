
The DVC folder link is below. Note it is not meant to be interacted with manually but only through DVC.

[AlphaPhage DVC Google Drive Folder](https://drive.google.com/drive/folders/14Lq-827oeazte3pGJPGwC4CY0e4jSYf4?usp=drive_link)


# Workflow for a New Repo User
## 1️⃣ Clone the repo
```
git clone <repo-url>
cd <repo>
```

## 2️⃣ Install DVC with Google Drive support

If using conda:
```
conda create -n myenv python=3.10
conda activate myenv
conda install -c conda-forge dvc-gdrive
```
# or if you have environment.yml: `conda env create -f environment.yml`

Must have dvc-gdrive installed to access Google Drive.

3️⃣ Generate Google Drive OAuth Credentials ☁️Before you can authenticate, you need to create the application credentials (Client ID and Secret) used for DVC's OAuth flow.Access Google Cloud Console: Navigate to the [alphaphage-dvc] project link provided above: https://console.cloud.google.com/apis/api/drive.googleapis.com/metrics?project=alphaphage-dvc.Go to Credentials: In the left navigation menu, go to APIs & Services $\rightarrow$ Credentials.Create OAuth Client ID:Click + CREATE CREDENTIALS and select OAuth client ID.Set Application type to Desktop app.Give it a memorable name (e.g., dvc-dev-machine).Click CREATE.Copy Credentials: A modal window will appear displaying your new Client ID and Client Secret.Update Config: Copy both values and paste them into your local config file, replacing the placeholders:Ini, TOML# In .dvc/config.local
['remote "gdrive_remote"']
    gdrive_client_id = <YOUR_CLIENT_ID>
    gdrive_client_secret = <YOUR_CLIENT_SECRET>
This step only needs to be done once per DVC project/team and provides the shared application credentials. The subsequent "Authenticate with Google OAuth" step uses these to generate the unique per-user token.

## 3️⃣ Create a local config for OAuth credentials

Since _.dvc/config.local_ is not committed, each user needs one.

Copy the template:
```
cp .dvc/config.local.template .dvc/config.local
```

Edit _.dvc/config.local_:
```
['remote "gdrive_remote"']
    gdrive_client_id = <YOUR_CLIENT_ID>
    gdrive_client_secret = <YOUR_CLIENT_SECRET>
```

Use the client ID/secret for the OAuth project you’ve enabled:
[alphaphage-dvc](https://console.cloud.google.com/apis/api/drive.googleapis.com/metrics?project=alphaphage-dvc)

These are per-user.

_.dvc/config.local_ must not be committed.

## 4️⃣ Authenticate with Google OAuth
`dvc pull`


Browser will open asking you to log in.

Log in with your Google account.

Google may ask for permissions; if the OAuth project is already verified/enabled, it should succeed.

After successful login, DVC caches the token locally (~~/.config/dvc/cred.json) → no need to repeat.

Each user must authenticate once per machine.

## 5️⃣ Pull the remote data
`dvc pull`


Downloads all tracked files from the Google Drive remote into the local workspace.

If the remote is empty (first push hasn’t been done), nothing will download yet.

## 6️⃣ Subsequent runs

After dvc pull succeeds once, users can just run dvc pull or dvc push normally.

The OAuth token persists locally; no need to re-enable the project for each user, only for their own credentials.

**Notes**

Each user needs their own OAuth credentials unless you switch to a service account (preferred for shared repos/CI).

If using a personal OAuth client, they do not need to enable the project — you already did that. They just need a client ID/secret added in their _.dvc/config.local_.

_.dvc/config.local_ ensures that credentials remain private per user and are ignored by Git.

# Adding data to DVC
To add data that is stored outside of the repo, use the `--external` option:
```
mkdir -p ml_assets/data/sequence_data/bacteria
dvc import-url --to-remote ../databases/production/sequence_data/bacteria* ml_assets/data/sequence_data  # for remote
# dvc add ../databases/production/sequence_data/bacteria --out ml_assets/data/sequence_data # for duplicated

git add ml_assets/data/sequence_data/bacteria/.gitignore ml_assets/data/sequence_data/bacteria/bacteria.dvc
git add ml_assets/data/sequence_data/bacteria.dvc ml_assets/data/sequence_data/.gitignore
dvc config core.autostage true # auto stage .dvc, dvc.yaml
for mistakes:
dvc remove ml_assets/data/sequence_data/bacteria.dvc

# Phage sources
dvc import-url --to-remote ../databases/production/sequence_data/phage ml_assets/data/sequence_data

dvc upda

```
Setting up the local cache:
```
dvc config cache.dir ../databases/production
dvc config cache.type symlink
```





# Initial setup for new repo
The generic instruction from ChatGPT are below:

# ✅ Step-by-Step: Configure DVC to Use Google Drive
1. Install DVC with GDrive Support
    ```
    pip install "dvc[gdrive]"
    ```
    This has been added to the conda environment files in _conda_envs/<environment name>.yml_.

    And then initialise dvc:
    ```
    dvc init
    ```
2. Create a Folder on Google Drive for DVC

You need a dedicated folder that will act as the remote storage.

Go to **Google Drive** → **Create a folder** (e.g., dvc-storage)

**Right-click** → **Get link** → **Set permissions:**

If collaborating: Anyone with link can view

If private: keep restricted (you'll authenticate per machine)

Copy the folder ID from the URL:

Example URL:

https://drive.google.com/drive/folders/14Lq-827oeazte3pGJPGwC4CY0e4jSYf4


Folder ID:

14Lq-827oeazte3pGJPGwC4CY0e4jSYf4

1. Add the Remote to DVC

    Use the folder ID:
    ```
    dvc remote add -d gdrive_remote gdrive://14Lq-827oeazte3pGJPGwC4CY0e4jSYf4
    ```

    Flags:

    `-d` marks this as the default remote.

    gdrive_remote is the remote name (choose any).

2. Configure Credentials (OAuth)

DVC will need OAuth to access Drive.

Run:

dvc pull


On first run DVC will print a URL, like:

Go to this URL to authorize DVC: https://accounts.google.com/o/oauth2/auth?...


Follow the link:

Choose the Google account

Approve access (only needed once per machine)

Copy the token returned

Paste it into the terminal

DVC stores the credentials in:

~/.config/dvc/cred.json


You should not commit this file.
## Auth "blocked app" problem
### Option 1 — Use your own Google OAuth client (recommended for production/collaborators)

               This is the proper way if you want to avoid the “blocked app” problem.

               Go to the Google Cloud Console
               .

               Create a new project (or use an existing one).

               Go to APIs & Services → OAuth consent screen:

               User Type: External (or Internal if your organization allows)

               Add your email as a test user

               Add the scope: https://www.googleapis.com/auth/drive (or leave default)

               Go to Credentials → Create Credentials → OAuth Client ID → Desktop App.

               Download the client_secret.json.

               1️⃣ Create your local config file

                Inside your repo root, create a file called:

                .dvc/config.local


                .config.local is automatically ignored by Git if you haven’t changed .gitignore (DVC’s default .gitignore already ignores it).

                You can confirm:

                cat .gitignore

                2️⃣ Add the service account path in the local config

                Open .dvc/config.local and add:

                ['remote "gdrive_remote"']
                    gdrive_service_account_json_file = /absolute/path/to/service-account-key.json


                Use the absolute path to your service account JSON.

                Replace gdrive_remote with the name of your DVC remote if different.

                This overrides the same setting in .dvc/config without affecting other users.

                3️⃣ Make sure Git ignores it

                .dvc/config.local should already be ignored by DVC’s default .gitignore. If not, add:

                .dvc/config.local


                This prevents committing secrets.

                4️⃣ Test the setup

                Run:

                dvc pull


                DVC will now use the service account key instead of asking for OAuth.

                You should no longer see the Google “blocked app” message.

                Works in CI/CD, headless servers, and other users on the repo (as long as the JSON is accessible locally).

                5️⃣ Optional: Verify in config

                Check which file DVC is using:

                dvc remote list
                dvc remote modify --list


                You should see:

                gdrive_remote:
                    gdrive_service_account_json_file = /absolute/path/to/service-account-key.json
### Option 2 — Use a Service Account (recommended for automation / CI/CD)

               Create a service account in Google Cloud.

               Share the Google Drive folder with the service account’s email.

               Download the JSON key.

               Tell DVC to use it:

               dvc remote modify gdrive_remote gdrive_service_account_json_file /path/to/key.json


               This avoids OAuth entirely. Perfect for headless environments like CI or multiple developers.

5. Update .dvc/config (automatically created)

This is how it normally looks:

['remote "gdrive_remote"']
    url = gdrive://1AbCdEfGhijklmnopQRstuVWxyz


You don’t need to manually edit anything unless you want custom caching/parallelism.

6. Push Data to Google Drive

Once remotes are set up:

dvc push


This uploads tracked files to your GDrive folder.

7. Testing Pull on a Fresh Machine

Clone your repo somewhere clean:

git clone <repo>
cd <repo>
pip install "dvc[gdrive]"
dvc pull


You’ll get the OAuth prompt again—this is expected unless you share credentials manually (not recommended).

# ⚠️ Common Pitfalls (and fixes)
## ❌ Remote folder is empty / inconsistent

Google Drive does not behave like S3—files are indexed by Drive file ID, not by path.
Let DVC manage that folder exclusively.

## ❌ Error: “File not found on remote”

Often means:

Google Drive changed permission settings

Someone renamed the folder

The OAuth token expired

Run:

dvc doctor
dvc pull -f

## ❌ GDrive "Quota exceeded"

This is common if you store many files because Drive has:

A daily upload quota

A MIME-type conversion overhead

Workaround: batch data in fewer files (e.g., use tarballs before tracking).

## ⭐ Optional Enhancements
1. Use a separate remote for models
```
dvc remote add models gdrive://<id>
dvc remote default models
```
1. Encrypt your credentials
dvc config remote.gdrive_remote.gdrive_use_service_account true


For service-account setups (more advanced).

3. Outsource large files through Google Drive Shared Drives

Shared Drives allow larger quotas and better access control.

## 🎉 Summary

Minimum Required Setup

pip install dvc[gdrive]

Create Drive folder → copy ID

dvc remote add -d gdrive_remote gdrive://<folder-id>

dvc push → authenticate → success

That's all you need for a full GDrive-backed DVC setup.
