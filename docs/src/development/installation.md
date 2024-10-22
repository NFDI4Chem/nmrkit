# Local development instructions

1. Clone the project from [GitHub](https://github.com/NFDI4Chem/nmrkit)

3. Open a terminal or command prompt.

4. Navigate to the desired directory: Use `cd` to navigate to the directory where you want to clone the project.

5. Clone the repository: Run the command `git clone https://github.com/NFDI4Chem/nmrkit.git` to clone the project.

6. Use `cd` to navigate into the cloned project directory.

You have successfully cloned the project from GitHub onto your local machine.

Once cloned you can either choose to run the project via Docker-compose (recommended) or locally (need to make sure you have all the dependencies resolved).

## Docker

1. Install Docker: Install [Docker](https://img.docker.com/get-docker/) on your machine by following the instructions for your specific operating system.

2. Use `cd` to navigate into the cloned project directory and create a .env file which can be copied from .env.template and provide your own values for password and username.

3. You can use the docker-compose.yml file in the root directory, which is same as below and update accordingly if required.

```yaml
version: "3.8"

services:
  web:
    build:
      context: ./
      dockerfile: Dockerfile
    container_name: nmrkit-api
    volumes:
      - ./app:/code/app
    ports:
      - "80:80"
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:80/latest/ping"]
      interval: 1m30s
      timeout: 10s
      retries: 20
      start_period: 60s
    env_file:
      - ./.env
  prometheus:
    image: prom/prometheus
    container_name: nmrkit_prometheus
    ports:
      - 9090:9090
    volumes:
      - ./prometheus_data/prometheus.yml:/etc/prometheus/prometheus.yml
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
  grafana:
    image: grafana/grafana
    container_name: nmrkit_grafana
    ports:
      - 3000:3000
    volumes:
      - grafana_data:/var/lib/grafana
  redis:
    image: "redis:alpine"
    ports:
        - "${FORWARD_REDIS_PORT:-6379}:6379"
    volumes:
        - "redis:/data"
    networks:
        - default
    healthcheck:
        test: ["CMD", "redis-cli", "ping"]
        retries: 3
        timeout: 5s
  pgsql:
    image: "informaticsmatters/rdkit-cartridge-debian"
    ports:
      - "${FORWARD_DB_PORT:-5432}:5432"
    env_file:
      - ./.env
    volumes:
      - "pgsql:/var/lib/postgresql/data"
    networks:
      - default
    healthcheck:
      test:
        [
            "CMD",
            "pg_isready",
            "-q",
            "-d",
            "${POSTGRES_DB}",
            "-U",
            "${POSTGRES_USER}",
        ]
      retries: 3
      timeout: 5s
  minio:
    image: 'minio/minio:latest'
    ports:
        - '${FORWARD_MINIO_PORT:-9001}:9001'
        - '${FORWARD_MINIO_CONSOLE_PORT:-8900}:8900'
    environment:
        MINIO_ROOT_USER: 'sail'
        MINIO_ROOT_PASSWORD: 'password'
    volumes:
        - 'minio:/data/minio'
    networks:
        - default
    command: minio server /data/minio --console-address ":8900"
    healthcheck:
        test: ["CMD", "curl", "-f", "http://localhost:9001/minio/health/live"]
        retries: 3
        timeout: 5s
volumes:
  prometheus_data:
    driver: local
    driver_opts:
      o: bind
      type: none
      device: ./prometheus_data
  grafana_data:
    driver: local
    driver_opts:
      o: bind
      type: none
      device: ./grafana_data
  redis:
    driver: local
  minio:
    driver: local
  pgsql:
    driver: local
networks:
  default: 
    name: nmrkit_vpc
```

4. Run Docker Compose: Execute the command ```docker-compose up -d``` to start the containers defined in the Compose file.

5. Wait for the containers to start: Docker Compose will start the containers and display their logs in the terminal or command prompt.

Unicorn will start the app and display the server address (usually `http://localhost:80`) and Grafana dashboard can be accessed at `http://localhost:3000`

You may update the docker-compose file to disable or add additional services but by default, the docker-compose file shipped with the project has the web (nmrkit FAST API app), [rdkit-cartridge-debian](https://hub.docker.com/r/informaticsmatters/rdkit-cartridge-debian), [Prometheus](https://prometheus.io/img/introduction/overview/) and [Grafana](https://prometheus.io/img/introduction/overview/) (logging and visualisation of metrics), [Minio](https://min.io/img/minio/linux/index.html), [Redis](https://redis.io/img/) services.

## Standalone

1. Install Python: Install Python on your machine by following the instructions for your specific operating system.

2. Open a terminal or command prompt.

3. Navigate to the directory where your CPM project codebase is located: Use `cd` to navigate to the project directory.

5. Create a virtual environment (optional but recommended): Run the command `python -m venv env` to create a new virtual environment named "env" for your app.

6. Activate the virtual environment (if created): Depending on your operating system, run the appropriate command to activate the virtual environment. For example, on Windows, run `.\env\Scripts\activate`, and on macOS/Linux, run `source env/bin/activate`.

7. Install FastAPI and required dependencies: Run the command `pip install -r requirements.txt` to install FastAPI and the necessary dependencies.

8. Run the FastAPI app: Execute the command `uvicorn main:app --reload` to start the CPM app.

9. Wait for the app to start: Uvicorn will start the app and display the server address (usually `http://localhost:8000`) in the terminal or command prompt.

10. Access the FastAPI app: Open a web browser and navigate to the server address displayed in the terminal or command prompt. You should see your FastAPI app running.

That's it!

## Workers

Uvicorn also has the option to start and run several worker processes.

Nevertheless, as of now, Uvicorn's capabilities for handling worker processes are more limited than Gunicorn's. So, if you want to have a process manager at this level (at the Python level), then it might be better to try Gunicorn as the process manager.

In any case, you would run it like this:

<div class="termy">

```console
$ uvicorn main:app --host 0.0.0.0 --port 8080 --workers 4
```

Update the Dockerfile in case you are running via docker-compose and rebuild the image for the changes to reflect.

```
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80", "--workers", "4"]
```

</div>


## Logging (Prometheus and  Grafana)

::: info

The following instructions are based on the blog post - https://dev.to/ken_mwaura1/getting-started-monitoring-a-fastapi-app-with-Grafana-and-Prometheus-a-step-by-step-guide-3fbn

To learn more about using Grafana in general, see the official [Prometheus](https://prometheus.io/img/introduction/overview/) and [Grafana](https://grafana.com/img/) documentation, or check out our other monitoring tutorials.

:::


Prometheus and Grafana are useful tools to monitor and visualize metrics in FastAPI applications.

Prometheus is a powerful monitoring system that collects and stores time-series data. By instrumenting your FastAPI app with Prometheus, you can collect various metrics such as request count, response time, error rate, and resource utilization. Grafana is a popular data visualization tool that integrates seamlessly with Prometheus. It allows you to create custom dashboards and visualize the collected metrics in a meaningful and interactive way. With Grafana, you can build visual representations of your FastAPI app's performance, monitor trends, and gain insights into your application's behaviour.

CPM docker-compose file comes prepackaged with Prometheus and Grafana services for you. When you run the docker-compose file these services also spin up automatically and will be available for you to monitor your application performance.

When you install CPM for the first time you need to configure your Prometheus source and enable it as the Grafana data source. You can then use the data source to create dashboards.

### Grafana Dashboard
Now that we have Prometheus running we can create a Grafana dashboard to visualize the metrics from our FastAPI app. To create a Grafana dashboard we need to do the following:

1. Create a new Grafana dashboard.
2. Add a new Prometheus data source.
3. Add a new graph panel.
4. Add a new query to the graph panel.
5. Apply the changes to the graph panel.
6. Save the dashboard.
7. View the dashboard.
8. Repeat steps 3-7 for each metric you want to visualize.
9. Repeat steps 2-8 for each dashboard you want to create.
10. Repeat steps 1-9 for each app you want to monitor.

Once you have Grafana running go to: localhost:3000. You should see the following screen:

Grafana login

Enter the default username and password (admin/admin) and click "Log In". You should be prompted to change the password. Enter a new password and click "Save". You should see the following screen:

<p align="center">
  <img align="center" src="/img/grafana_login.jpeg" alt="Logo" style="filter: drop-shadow(0px 0px 10px rgba(0, 0, 0, 0.5));" width="auto">
</p>

Grafana home

Click on the "Create your first data source" button. You should see the following screen:

<p align="center">
  <img align="center" src="/img/grafana.png" alt="Logo" style="filter: drop-shadow(0px 0px 10px rgba(0, 0, 0, 0.5));" width="auto">
</p>

Grafana add the data source

<p align="center">
  <img align="center" src="/img/grafana_ds.png" alt="Logo" style="filter: drop-shadow(0px 0px 10px rgba(0, 0, 0, 0.5));" width="auto">
</p>


Click on the "Prometheus" button. You should see the following screen:

<p align="center">
  <img align="center" src="/img/prometheus.png" alt="Logo" style="filter: drop-shadow(0px 0px 10px rgba(0, 0, 0, 0.5));" width="auto">
</p>

Enter the following information:

Name: Prometheus <br/>
URL: http://Prometheus:9090 <br/>
Access: Server (Default) <br/>
Scrape interval: 15s <br/>
HTTP Method: GET <br/>
HTTP Auth: None <br/>
Basic Auth: None <br/>
With Credentials: No <br/>
TLS Client Auth: None <br/>
TLS CA Certificate: None <br/>

Click on the "Save & Test" button. You should see the following screen:

<p align="center">
  <img align="center" src="/img/grafana_ds_saved.png" alt="Logo" style="filter: drop-shadow(0px 0px 10px rgba(0, 0, 0, 0.5));" width="auto">
</p>

Click on the "Dashboards" button. You should see the following screen:

<p align="center">
  <img align="center" src="/img/grafana_db.png" alt="Logo" style="filter: drop-shadow(0px 0px 10px rgba(0, 0, 0, 0.5));" width="auto">
</p>

Click on the ""New Dashboard" button. You should see the following screen:

<p align="center">
  <img align="center" src="/img/grafana_db_new.png" alt="Logo" style="filter: drop-shadow(0px 0px 10px rgba(0, 0, 0, 0.5));" width="auto">
</p>

Download the NMRKit dashboard template (JSON) here - https://github.com/NFDI4Chem/nmrkit/blob/main/api-dashboard.json

## Benchmarking / Stress testing

[Vegeta](https://github.com/tsenart/vegeta) is an open-source command-line tool written in Go, primarily used for load testing and benchmarking HTTP services. It allows you to simulate a high volume of requests to a target URL and measure the performance characteristics of the service under various loads.


To perform stress testing using Vegeta, you can follow these steps:

1. Install Vegeta: Start by installing Vegeta on your machine. You can download the latest release binary from the official GitHub repository (https://github.com/tsenart/vegeta) or use a package manager like Homebrew (macOS/Linux) or Chocolatey (Windows) for installation.

2. Define a target endpoint: Identify the specific FastAPI endpoint you want to stress test. Make sure you have the URL and any necessary authentication or headers required to access the endpoint.

3. Prepare a Vegeta target file: Create a text file, e.g., `target.txt`, and define the target URL using the Vegeta target format. For example:

   ```plaintext
   GET http://localhost:8000/my-endpoint
   ```

   Replace `http://localhost:8000/my-endpoint` with the actual URL of your FastAPI endpoint.

4. Create a Vegeta attack plan: In another text file, e.g., `attack.txt`, specify the rate and duration for the stress test using the Vegeta attack format. For example:

   ```plaintext
   rate: 100
   duration: 10s
   ```

   This example sets the request rate to 100 requests per second for a duration of 10 seconds. Adjust the values according to your requirements.

5. Run the Vegeta attack: Open a terminal or command prompt, navigate to the directory where the target and attack files are located, and execute the following command:

   ```bash
   vegeta attack -targets=target.txt -rate=attack.txt | vegeta report
   ```

   This command runs the Vegeta attack using the target and attack files, sends requests to the specified FastAPI endpoint, and generates a report with statistics.

6. Analyze the stress test results: Vegeta will output detailed metrics and performance statistics for the stress test. It includes data on request rate, latency, success rate, and more. Analyze these results to evaluate the performance and stability of your FastAPI application under stress.

By following these steps, you can perform stress testing on your CPM FASTAPI application using Vegeta, generating load and analyzing the performance characteristics of your endpoints. This process helps identify potential bottlenecks and validate the scalability of your application.

## Linting / Formatting

We recommend using flake8 and Black to perform linting and formatting in Python

1. Install flake8 and Black: Start by installing both flake8 and Black. You can install them using pip by running the following command:
   
   ```bash
   pip install flake8 black
   ```

2. Linting with flake8: flake8 is a popular Python linter that checks your code for style and potential errors. Run flake8 by executing the following command in your project directory:

   ```bash
   flake8 --per-file-ignores="__init__.py:F401" --ignore E402,E501,W503 $(git ls-files '*.py') .
   ```

   flake8 will analyze your code and provide feedback on any style violations or issues found.

3. Formatting with Black: Black is a Python code formatter that enforces a consistent style and automatically formats your code. To format your code with Black, run the following command in your project directory:

   ```bash
   black .
   ```

   or 

   ```bash
    black $(git ls-files '*.py') .
   ```

   The `.` specifies the current directory. Black will recursively format all Python files within the directory and apply the necessary formatting changes.

   Note: Black uses a strict formatting style, so it's a good practice to make sure you have committed your changes to a version control system before running Black.