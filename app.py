from openai import OpenAI
import seaborn as sns
import pandas as pd
from flask import Flask, render_template, request, jsonify
import json
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

# Create Flask application instance
app = Flask(__name__)

# Load OpenAI credentials
with open('/workspaces/summerschool/credentials.json', 'r', encoding='utf-8') as f:
    credentials = json.load(f)

# Initialize OpenAI client
client = OpenAI(api_key=credentials['openai']['api_key'])

# Routes


@app.route('/')
def index():
    """Simple home page route for testing Flask webapp"""
    return render_template('index.html')


@app.route('/analyze', methods=['POST'])
def analyze_question():
    """Process user question and generate chart"""
    question = request.form.get('question', '').strip()

    if not question:
        return jsonify({'error': 'Please enter a question'})

    try:
        # Load the shark attack dataset
        df = pd.read_csv(
            '/workspaces/summerschool/Python_Multi-Agent_Augmented_Analytics/data/global-shark-attack.csv')

        # Generate code using OpenAI
        chart_code = generate_chart_code(question, df)

        # Execute the generated code and create chart
        success = execute_chart_code(chart_code, df)

        if success:
            return jsonify({
                'success': True,
                'question': question
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Chart generation failed - check the saved error image'
            })

    except Exception as e:
        return jsonify({'error': f'Error generating chart: {str(e)}'})


def generate_chart_code(question, df):
    """Use OpenAI to generate matplotlib code based on the question"""

    # Get basic info about the dataset
    columns_info = []
    for col in df.columns[:10]:  # Limit to first 10 columns for prompt size
        dtype = str(df[col].dtype)
        sample_values = df[col].dropna().head(3).tolist()
        columns_info.append(f"- {col} ({dtype}): {sample_values}")

    dataset_info = "\n".join(columns_info)

    prompt = f"""
You are a data visualization expert. Generate Python matplotlib code to answer this question about shark attack data:

Question: "{question}"

Dataset columns available:
{dataset_info}

The dataframe is already loaded as 'df'. Generate clean, efficient matplotlib code that:
1. Creates a chart to answer the question
2. Uses appropriate chart type (bar, line, pie, histogram, etc.)
3. Includes proper title, labels, and formatting
4. Handles missing data appropriately
5. Returns a matplotlib figure object stored in variable 'fig'

Only provide the Python code, no explanations. The code should be executable as-is.

Example format:
```python
import matplotlib.pyplot as plt
import pandas as pd

# Your analysis code here
fig, ax = plt.subplots(figsize=(10, 6))
# Chart creation code
plt.title("Chart Title")
plt.tight_layout()
```
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # Using available model instead of gpt-4.1-nano
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1000,
            temperature=0.3
        )

        code = response.choices[0].message.content

        # Extract code from markdown if present
        if "```python" in code:
            code = code.split("```python")[1].split("```")[0].strip()
        elif "```" in code:
            code = code.split("```")[1].split("```")[0].strip()

        return code

    except Exception as e:
        raise Exception(f"OpenAI API error: {str(e)}")


def execute_chart_code(code, df):
    """Execute the generated code and return base64 encoded chart"""

    # Create a safe execution environment
    exec_globals = {
        'plt': plt,
        'pd': pd,
        'df': df,
        'sns': sns,
        'matplotlib': matplotlib
    }

    try:
        # Execute the generated code
        exec(code, exec_globals)

        # Get the figure object
        fig = exec_globals.get('fig', plt.gcf())

        # Save the chart as graphic00.png
        chart_path = '/workspaces/summerschool/static/graphic00.png'
        fig.savefig(chart_path, format='png', dpi=150, bbox_inches='tight')

        # Clean up
        plt.close(fig)

        return True

    except Exception as e:
        # Create an error chart if code execution fails
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, f'Error creating chart:\n{str(e)}',
                ha='center', va='center', fontsize=12,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral"))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        plt.title('Chart Generation Error')

        # Save the error chart
        chart_path = '/workspaces/summerschool/static/graphic00.png'
        fig.savefig(chart_path, format='png', dpi=150, bbox_inches='tight')

        plt.close(fig)

        return False


if __name__ == '__main__':
    # Run the application
    app.run(host='0.0.0.0', port=5000, debug=True)
