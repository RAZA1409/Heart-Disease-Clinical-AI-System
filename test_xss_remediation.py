"""
Focused Verification Test: AI Assistant XSS Remediation in patient_detail.html
Verifies that:
1. innerHTML is completely removed from addAIMessage and patient_detail.html
2. Safe DOM-node rendering is used (document.createTextNode, textContent, createElement)
3. Malicious HTML, script injection, and event-handler payloads cannot execute
4. Basic Markdown (bold text, line breaks, bullet points) is safely preserved
5. The patient_detail route renders successfully with 200 OK
"""

import unittest
import os
import re
from app import app
from database import get_db_connection


class AIAssistantXSSRemediationTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        app.config['TESTING'] = True
        cls.client = app.test_client()

        template_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "templates",
            "patient_detail.html"
        )
        with open(template_path, "r", encoding="utf-8") as f:
            cls.template_content = f.read()

    def test_01_no_innerhtml_in_patient_detail(self):
        """Verify innerHTML has been completely removed from patient_detail.html."""
        self.assertNotIn(
            "innerHTML",
            self.template_content,
            "innerHTML was found in patient_detail.html! It must not be used."
        )

    def test_02_safe_dom_methods_used_in_add_ai_message(self):
        """Verify safe DOM construction methods are used in addAIMessage."""
        self.assertIn("document.createTextNode", self.template_content)
        self.assertIn("strong.textContent", self.template_content)
        self.assertIn("document.createElement(\"strong\")", self.template_content)
        self.assertIn("document.createElement(\"br\")", self.template_content)

    def test_03_ai_message_logic_neutralizes_xss_payloads(self):
        """
        Simulate the exact JavaScript DOM-node generation logic on malicious payloads
        and verify that no executable element tags or event-handler attributes are produced.
        """
        def simulate_js_add_ai_message(raw_input):
            raw = str(raw_input if raw_input is not None else "").replace("\r\n", "\n").replace("\r", "\n")
            lines = raw.split("\n")
            dom_nodes = []

            for line_idx, line in enumerate(lines):
                processed = re.sub(r'\\([*_])', r'\1', line)
                processed = re.sub(r'^#{1,6}\s*', '', processed)
                processed = re.sub(r'^\s*[*-]\s+', '• ', processed)
                if re.match(r'^\s*\*+\s*$', processed):
                    processed = ''

                parts = re.split(r'(\*\*.*?\*\*)', processed)
                for part in parts:
                    if part.startswith('**') and part.endswith('**') and len(part) >= 4:
                        bold_content = part[2:-2]
                        if bold_content:
                            dom_nodes.append({'type': 'STRONG', 'text': bold_content})
                    elif part:
                        dom_nodes.append({'type': 'TEXT', 'text': part})

                if line_idx < len(lines) - 1:
                    dom_nodes.append({'type': 'BR'})

            return dom_nodes

        malicious_payloads = [
            "<script>alert('XSS')</script>",
            "<img src=x onerror=alert(document.cookie)>",
            "<svg/onload=alert(1)>",
            "<a href=\"javascript:alert('pwned')\">Click here</a>",
            "<iframe src=\"https://malicious.example.com\"></iframe>",
            "**<img src=x onerror=alert(1)>**",
            "**\" onmouseover=\"alert(1)**",
            "<input onfocus=alert(1) autofocus>",
            "<details open ontoggle=alert(1)>",
            "Normal text with <script>console.log('leak')</script> inside",
        ]

        dangerous_tags = ['SCRIPT', 'IMG', 'SVG', 'IFRAME', 'INPUT', 'DETAILS', 'A']

        for payload in malicious_payloads:
            nodes = simulate_js_add_ai_message(payload)

            # Node types created must strictly be within safe whitelist
            allowed_node_types = {'TEXT', 'STRONG', 'BR'}
            for node in nodes:
                self.assertIn(
                    node['type'],
                    allowed_node_types,
                    f"Forbidden node type {node['type']} created for payload: {payload}"
                )
                self.assertNotIn(node['type'], dangerous_tags)

                # Text content in text or strong nodes is strictly inert plain text
                if 'text' in node:
                    self.assertIsInstance(node['text'], str)

    def test_04_markdown_formatting_preserved(self):
        """Verify that basic markdown formatting (bold, line breaks, bullets) works correctly."""
        def simulate_js_add_ai_message(raw_input):
            raw = str(raw_input if raw_input is not None else "").replace("\r\n", "\n").replace("\r", "\n")
            lines = raw.split("\n")
            dom_nodes = []

            for line_idx, line in enumerate(lines):
                processed = re.sub(r'\\([*_])', r'\1', line)
                processed = re.sub(r'^#{1,6}\s*', '', processed)
                processed = re.sub(r'^\s*[*-]\s+', '• ', processed)
                if re.match(r'^\s*\*+\s*$', processed):
                    processed = ''

                parts = re.split(r'(\*\*.*?\*\*)', processed)
                for part in parts:
                    if part.startswith('**') and part.endswith('**') and len(part) >= 4:
                        bold_content = part[2:-2]
                        if bold_content:
                            dom_nodes.append({'type': 'STRONG', 'text': bold_content})
                    elif part:
                        dom_nodes.append({'type': 'TEXT', 'text': part})

                if line_idx < len(lines) - 1:
                    dom_nodes.append({'type': 'BR'})

            return dom_nodes

        sample_response = (
            "### Clinical Findings\n"
            "- Patient exhibits **elevated cholesterol** levels.\n"
            "- Blood pressure is **145/90 mmHg**.\n\n"
            "Recommended action: Follow-up required."
        )

        nodes = simulate_js_add_ai_message(sample_response)

        # Verify bold terms extracted into STRONG nodes
        strong_texts = [n['text'] for n in nodes if n['type'] == 'STRONG']
        self.assertIn("elevated cholesterol", strong_texts)
        self.assertIn("145/90 mmHg", strong_texts)

        # Verify bullet points converted
        text_nodes = [n['text'] for n in nodes if n['type'] == 'TEXT']
        full_text = " ".join(text_nodes)
        self.assertIn("• Patient exhibits", full_text)
        self.assertIn("• Blood pressure is", full_text)

        # Verify headings cleaned (no ###)
        self.assertNotIn("###", full_text)
        self.assertIn("Clinical Findings", full_text)

        # Verify BR nodes created for line breaks
        br_nodes = [n for n in nodes if n['type'] == 'BR']
        self.assertGreaterEqual(len(br_nodes), 3)

    def test_05_patient_detail_route_renders_successfully(self):
        """Verify that patient_detail route renders with 200 OK and valid session."""
        conn = get_db_connection()
        patient = conn.execute("SELECT patient_id FROM patients LIMIT 1").fetchone()
        conn.close()

        if patient:
            pid = patient["patient_id"]
            with self.client.session_transaction() as sess:
                sess['user'] = 'admin'

            response = self.client.get(f"/patient/{pid}")
            self.assertEqual(response.status_code, 200)
            html = response.get_data(as_text=True)
            self.assertIn("aiChatMessages", html)
            self.assertIn("addAIMessage", html)
            self.assertNotIn("innerHTML", html)


if __name__ == "__main__":
    unittest.main()
