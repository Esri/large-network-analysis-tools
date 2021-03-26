The unit tests use data and from the SanFrancisco.gdb geodatabase from the ArcGIS Pro Network Analyst tutorial data. Download the data from https://links.esri.com/NetworkAnalyst/TutorialData/Pro. Extract the zip file and put SanFrancisco.gdb here in a folder called "TestInput".

The tests also require a file called portal_credentials.py with the following variables:
portal_url = "<url to your portal>"
portal_username = "<your username>"
portal_password = "<your password>"
portal_travel_mode = "<your travel mode name>"
This file is ignored by the GitHub repo but is required to successfully run the tests.