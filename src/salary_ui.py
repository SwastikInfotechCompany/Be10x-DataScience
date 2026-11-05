"""
Industrial Data Analytics Unified UI Server
============================================
Lightweight zero-dependency web server built on Python standard library http.server.
Serves both Part 1 (Salary Prediction Audit) and Part 2 (Customer Segmentation Dashboard)
with live interactive REST API endpoints.

Usage:
  python src/salary_ui.py [--port 8000]
"""

import os
import sys
import json
import urllib.parse
import numpy as np
from http.server import HTTPServer, SimpleHTTPRequestHandler

PORT = 8000
REPORTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "reports"))
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
SCREENSHOTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "ui_screenshots"))

class IndustrialDashboardHandler(SimpleHTTPRequestHandler):
    def translate_path(self, path):
        clean_path = path.split('?', 1)[0].split('#', 1)[0]
        
        # Navigation routes
        if clean_path in ['/', '/index.html']:
            # Default to Part 2 or Part 1 if requested
            return os.path.join(REPORTS_DIR, 'customer_segmentation_dashboard.html')
        elif clean_path in ['/part1', '/salary']:
            return os.path.join(REPORTS_DIR, 'salary_prediction_dashboard.html')
        elif clean_path in ['/part2', '/segmentation', '/customers']:
            return os.path.join(REPORTS_DIR, 'customer_segmentation_dashboard.html')
        
        # Static files from reports
        target_rep = os.path.join(REPORTS_DIR, clean_path.lstrip('/'))
        if os.path.exists(target_rep):
            return target_rep
            
        # Static files from ui_screenshots
        target_ss = os.path.join(SCREENSHOTS_DIR, clean_path.lstrip('/'))
        if os.path.exists(target_ss):
            return target_ss
            
        # Static files from data
        target_data = os.path.join(DATA_DIR, clean_path.lstrip('/'))
        if os.path.exists(target_data):
            return target_data
            
        return super().translate_path(path)

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        
        # API 1: /api/part1/audit
        if parsed.path == '/api/part1/audit':
            fpath = os.path.join(REPORTS_DIR, 'salary_prediction_audit.json')
            self.serve_json_file(fpath)
            return

        # API 2: /api/part1/predict?experience=15.0
        elif parsed.path == '/api/part1/predict':
            query = urllib.parse.parse_qs(parsed.query)
            try:
                exp = float(query.get('experience', [15.0])[0])
            except ValueError:
                exp = 15.0
            
            fpath = os.path.join(REPORTS_DIR, 'salary_prediction_audit.json')
            if os.path.exists(fpath):
                with open(fpath, 'r', encoding='utf-8') as f:
                    audit = json.load(f)
                slope = audit['model_results']['slope_m']
                intercept = audit['model_results']['intercept_c']
                rmse = audit['model_results']['metrics']['rmse']
                
                pred_sal = (slope * exp) + intercept
                margin = 1.96 * rmse
                
                resp = {
                    'experience_years': exp,
                    'predicted_annual_salary': round(pred_sal, 2),
                    'predicted_monthly_salary': round(pred_sal / 12.0, 2),
                    'confidence_interval_95': {
                        'min': round(max(0, pred_sal - margin), 2),
                        'max': round(pred_sal + margin, 2)
                    },
                    'equation_used': f"Y = ({slope:,.2f} * {exp}) + {intercept:,.2f}"
                }
                self.send_json_response(resp)
            else:
                self.send_json_response({'error': 'Part 1 model not trained yet.'}, status=404)
            return

        # API 3: /api/part2/audit
        elif parsed.path in ['/api/part2/audit', '/api/segmentation/audit']:
            fpath = os.path.join(REPORTS_DIR, 'customer_segmentation_audit.json')
            self.serve_json_file(fpath)
            return

        # API 4: /api/part2/classify?age=25&spend=2500
        elif parsed.path in ['/api/part2/classify', '/api/segmentation/classify']:
            query = urllib.parse.parse_qs(parsed.query)
            try:
                age = float(query.get('age', [25.0])[0])
                spend = float(query.get('spend', [2500.0])[0])
            except ValueError:
                age, spend = 25.0, 2500.0
                
            fpath = os.path.join(REPORTS_DIR, 'customer_segmentation_audit.json')
            if os.path.exists(fpath):
                with open(fpath, 'r', encoding='utf-8') as f:
                    audit = json.load(f)
                    
                means = audit['scaling_audit']['means']
                stds = audit['scaling_audit']['stds']
                profiles = audit['cluster_profiles']
                
                # Scaled coordinates
                age_scaled = (age - means['age']) / stds['age']
                spend_scaled = (spend - means['annual_spend']) / stds['annual_spend']
                
                best_cluster = 0
                min_dist = float('inf')
                distances = {}
                
                for cid, prof in profiles.items():
                    c_scaled = prof['centroid_scaled']
                    d = float(np.sqrt((age_scaled - c_scaled[0])**2 + (spend_scaled - c_scaled[1])**2))
                    distances[cid] = round(d, 4)
                    if d < min_dist:
                        min_dist = d
                        best_cluster = int(cid)
                        
                assigned = profiles[str(best_cluster)]
                resp = {
                    'customer_input': {'age': age, 'annual_spend': spend},
                    'scaled_coordinates': [round(age_scaled, 4), round(spend_scaled, 4)],
                    'assigned_cluster': best_cluster,
                    'persona_name': assigned['persona_name'],
                    'loyalty_tier': assigned['loyalty_tier'],
                    'distance_to_centroid': round(min_dist, 4),
                    'distances_to_all_centroids': distances,
                    'recommended_discount': assigned['recommended_discount'],
                    'email_strategy': assigned['email_strategy']
                }
                self.send_json_response(resp)
            else:
                self.send_json_response({'error': 'Part 2 model not trained yet.'}, status=404)
            return

        super().do_GET()

    def serve_json_file(self, filepath):
        if os.path.exists(filepath):
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self.send_json_response(data)
        else:
            self.send_json_response({'error': f'File not found: {os.path.basename(filepath)}'}, status=404)

    def send_json_response(self, data, status=200):
        body = json.dumps(data, indent=2).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        sys.stderr.write(f"[Industrial Analytics Server] {self.address_string()} - {args[0]} {args[1]}\n")

def launch_ui_server(port=PORT, reports_dir=None):
    global REPORTS_DIR
    if reports_dir:
        REPORTS_DIR = os.path.abspath(reports_dir)
        
    server_address = ('', port)
    httpd = HTTPServer(server_address, IndustrialDashboardHandler)
    print("\n" + "=" * 75)
    print("INDUSTRIAL DATA ANALYTICS UNIFIED UI SERVER")
    print(f"Local Server URL: http://localhost:{port}")
    print(f"Part 1 Dashboard (Salary Prediction):     http://localhost:{port}/part1")
    print(f"Part 2 Dashboard (Customer Segmentation): http://localhost:{port}/part2")
    print(f"Part 1 API: http://localhost:{port}/api/part1/predict?experience=15")
    print(f"Part 2 API: http://localhost:{port}/api/part2/classify?age=25&spend=2500")
    print("Press Ctrl+C to stop the server.")
    print("=" * 75 + "\n")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down UI Server...")
        httpd.server_close()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Launch Unified Industrial Analytics UI Server")
    parser.add_argument('--port', type=int, default=PORT, help='Port to bind (default: 8000)')
    parser.add_argument('--reports-dir', type=str, default=REPORTS_DIR, help='Path to reports directory')
    args = parser.parse_args()
    
    launch_ui_server(port=args.port, reports_dir=args.reports_dir)
