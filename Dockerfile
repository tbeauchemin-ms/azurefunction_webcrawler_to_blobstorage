FROM mcr.microsoft.com/azure-functions/python:4-python3.11-appservice

ENV AzureWebJobsScriptRoot=/home/site/wwwroot AzureFunctionsJobHost__Logging__Console__IsEnabled=true WEBSITES_INCLUDE_CLOUD_CERTS=true

# Install system dependencies for Chromium / Playwright
RUN apt-get update && apt-get install -y wget gnupg curl libnss3 libatk-bridge2.0-0 libxss1 libasound2 libxshmfence1 libgbm1 libgtk-3-0 fonts-liberation libappindicator3-1 xdg-utils && rm -rf /var/lib/apt/lists/*

# Copy your function code and requirements
COPY . /home/site/wwwroot
WORKDIR /home/site/wwwroot

RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Install Playwright browser(s)
RUN playwright install --with-deps chromium

# EXPOSE 8000

# Explicitly set the entrypoint for Azure Functions Python worker
CMD [ "/azure-functions-host/Microsoft.Azure.WebJobs.Script.WebHost" ]

