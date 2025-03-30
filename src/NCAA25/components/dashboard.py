import os
import pandas as pd
import numpy as np
import json
import shutil
import matplotlib.pyplot as plt
import seaborn as sns
from src.NCAA25 import logger
from src.NCAA25.entity import DashboardConfig
from src.NCAA25.utils.common import load_model, load_dataframe, create_directories
from jinja2 import Environment, FileSystemLoader


class Dashboard:
    def __init__(self, config: DashboardConfig):
        self.config = config
        self.model = load_model(self.config.model_path) if os.path.exists(self.config.model_path) else None
        self.submission = pd.read_csv(self.config.submission_path) if os.path.exists(
            self.config.submission_path) else None

    def prepare_template_files(self):
        """Copy and prepare template files for the dashboard"""
        try:
            # Ensure directories exist
            create_directories([self.config.template_dir, self.config.static_dir])

            # Create index.html template
            index_template = """
            <!DOCTYPE html>
            <html>
            <head>
                <title>NCAA Tournament Predictions Dashboard</title>
                <link rel="stylesheet" href="{{ url_for('static', filename='css/bootstrap.min.css') }}">
                <link rel="stylesheet" href="{{ url_for('static', filename='css/dashboard.css') }}">
            </head>
            <body>
                <nav class="navbar navbar-dark bg-dark">
                    <div class="container-fluid">
                        <span class="navbar-brand mb-0 h1">NCAA Tournament Predictions Dashboard</span>
                    </div>
                </nav>

                <div class="container mt-4">
                    <div class="row">
                        <div class="col-md-12">
                            <div class="card">
                                <div class="card-header">
                                    <h5>Model Performance</h5>
                                </div>
                                <div class="card-body">
                                    <div class="row">
                                        <div class="col-md-6">
                                            <img src="{{ url_for('static', filename='images/roc_curve.png') }}" class="img-fluid">
                                        </div>
                                        <div class="col-md-6">
                                            <img src="{{ url_for('static', filename='images/precision_recall_curve.png') }}" class="img-fluid">
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>

                    <div class="row mt-4">
                        <div class="col-md-12">
                            <div class="card">
                                <div class="card-header">
                                    <h5>Feature Importance</h5>
                                </div>
                                <div class="card-body">
                                    <img src="{{ url_for('static', filename='images/feature_importance.png') }}" class="img-fluid">
                                </div>
                            </div>
                        </div>
                    </div>

                    <div class="row mt-4">
                        <div class="col-md-12">
                            <div class="card">
                                <div class="card-header">
                                    <h5>Upset Probabilities</h5>
                                </div>
                                <div class="card-body">
                                    <img src="{{ url_for('static', filename='images/upset_probability.png') }}" class="img-fluid">
                                </div>
                            </div>
                        </div>
                    </div>

                    <div class="row mt-4">
                        <div class="col-md-12">
                            <div class="card">
                                <div class="card-header">
                                    <h5>Tournament Bracket Simulator</h5>
                                </div>
                                <div class="card-body">
                                    <div class="bracket-container">
                                        <!-- Bracket visualization will be rendered here -->
                                        <div id="bracket"></div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>

                <script src="{{ url_for('static', filename='js/jquery.min.js') }}"></script>
                <script src="{{ url_for('static', filename='js/bootstrap.bundle.min.js') }}"></script>
                <script src="{{ url_for('static', filename='js/d3.min.js') }}"></script>
                <script src="{{ url_for('static', filename='js/dashboard.js') }}"></script>
            </body>
            </html>
            """

            # Write index template
            with open(os.path.join(self.config.template_dir, 'index.html'), 'w') as f:
                f.write(index_template)

            # Create CSS file
            os.makedirs(os.path.join(self.config.static_dir, 'css'), exist_ok=True)
            css_content = """
            .bracket-container {
                width: 100%;
                overflow-x: auto;
                margin-top: 20px;
            }

            .team {
                font-size: 12px;
                padding: 5px;
                border: 1px solid #ccc;
                margin: 2px 0;
                border-radius: 3px;
            }

            .winner {
                background-color: #d4edda;
                border-color: #c3e6cb;
            }

            .loser {
                background-color: #f8d7da;
                border-color: #f5c6cb;
            }
            """

            with open(os.path.join(self.config.static_dir, 'css', 'dashboard.css'), 'w') as f:
                f.write(css_content)

            # Create JavaScript file
            os.makedirs(os.path.join(self.config.static_dir, 'js'), exist_ok=True)
            js_content = """
            $(document).ready(function() {
                console.log('Dashboard initialized');

                // Load bracket data
                $.getJSON('/api/bracket', function(data) {
                    renderBracket(data);
                });
            });

            function renderBracket(data) {
                // Simple bracket rendering function
                const bracket = $('#bracket');
                bracket.empty();

                // For simplicity, just showing the bracket structure
                let html = '<div class="alert alert-info">Interactive bracket will be rendered here</div>';
                bracket.html(html);
            }
            """

            with open(os.path.join(self.config.static_dir, 'js', 'dashboard.js'), 'w') as f:
                f.write(js_content)

            # Copy analysis images to static directory
            os.makedirs(os.path.join(self.config.static_dir, 'images'), exist_ok=True)
            analysis_images = ['feature_importance.png', 'roc_curve.png',
                               'precision_recall_curve.png', 'upset_probability.png']

            for image in analysis_images:
                src_path = os.path.join(self.config.analysis_dir, 'reports', image)
                if os.path.exists(src_path):
                    shutil.copy(src_path, os.path.join(self.config.static_dir, 'images', image))

            logger.info("Prepared dashboard template files")

        except Exception as e:
            logger.error(f"Error preparing dashboard templates: {e}")
            raise e

    def prepare_api_data(self):
        """Prepare data for API endpoints"""
        try:
            api_data = {}

            # Load feature importance
            feat_imp_path = os.path.join(self.config.analysis_dir, 'reports', 'feature_importance.csv')
            if os.path.exists(feat_imp_path):
                feat_imp = pd.read_csv(feat_imp_path)
                api_data['feature_importance'] = feat_imp.to_dict(orient='records')

            # Load seed performance
            seed_perf_path = os.path.join(self.config.analysis_dir, 'reports', 'seed_performance.csv')
            if os.path.exists(seed_perf_path):
                seed_perf = pd.read_csv(seed_perf_path)
                api_data['seed_performance'] = seed_perf.to_dict(orient='records')

            # Load bracket simulation
            bracket_path = os.path.join(self.config.analysis_dir, 'reports', 'bracket_simulation.csv')
            if os.path.exists(bracket_path):
                bracket = pd.read_csv(bracket_path)
                api_data['bracket'] = bracket.to_dict(orient='records')

            # Save API data as JSON
            os.makedirs(os.path.join(self.config.static_dir, 'data'), exist_ok=True)
            with open(os.path.join(self.config.static_dir, 'data', 'api_data.json'), 'w') as f:
                json.dump(api_data, f)

            logger.info("Prepared API data for dashboard")

            return api_data
        except Exception as e:
            logger.error(f"Error preparing API data: {e}")
            raise e

    def create_app_script(self):
        """Create Flask application script"""
        try:
            app_script = """from flask import Flask, render_template, jsonify, send_from_directory
import os
import json

app = Flask(__name__, 
            template_folder='{template_dir}',
            static_folder='{static_dir}')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/feature_importance')
def feature_importance():
    with open(os.path.join(app.static_folder, 'data', 'api_data.json')) as f:
        data = json.load(f)
    return jsonify(data.get('feature_importance', []))

@app.route('/api/seed_performance')
def seed_performance():
    with open(os.path.join(app.static_folder, 'data', 'api_data.json')) as f:
        data = json.load(f)
    return jsonify(data.get('seed_performance', []))

@app.route('/api/bracket')
def bracket():
    with open(os.path.join(app.static_folder, 'data', 'api_data.json')) as f:
        data = json.load(f)
    return jsonify(data.get('bracket', []))

if __name__ == '__main__':
    app.run(host='{host}', port={port}, debug=True)
""".format(template_dir=self.config.template_dir,
           static_dir=self.config.static_dir,
           host=self.config.host,
           port=self.config.port)

            # Create app script file
            with open(os.path.join(self.config.root_dir, 'app.py'), 'w') as f:
                f.write(app_script)

            logger.info(f"Created Flask application script at {os.path.join(self.config.root_dir, 'app.py')}")

            return os.path.join(self.config.root_dir, 'app.py')
        except Exception as e:
            logger.error(f"Error creating app script: {e}")
            raise e

    def setup_dashboard(self):
        """Setup the complete dashboard"""
        try:
            # Prepare templates
            self.prepare_template_files()

            # Prepare API data
            self.prepare_api_data()

            # Create app script
            app_script_path = self.create_app_script()

            # Create a run script
            run_script = f"""
            # Run this script to start the dashboard
            python {app_script_path}
            """

            with open('run_dashboard.py', 'w') as f:
                f.write(run_script)

            logger.info("Dashboard setup completed")
            logger.info(f"To start the dashboard, run: python {app_script_path}")

            return app_script_path
        except Exception as e:
            logger.error(f"Error setting up dashboard: {e}")
            raise e