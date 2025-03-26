import os
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

import gradio as gr
import pandas as pd
from openai import OpenAI
from lxml import etree

# Initialize OpenAI client
client = OpenAI()

# Configuration - UPDATED SETTINGS
LLM_TEMPERATURE = 0.0
LLM_MODEL_ID = 'gpt-4'  # or 'gpt-3.5-turbo' for smaller context
LLM_MAX_OUT_TOKENS = 2000  # Reduced from 10000 to prevent overflow
MAX_ASSETS = 5  # Process only first 5 assets
MAX_SCENARIOS = 5  # Process only first 5 scenarios

def load_csv_data(input_file: str) -> gr.Dataframe:
    input_file = input_file.name
    print(f'Opening CSV: {input_file}')
    df = pd.read_csv(input_file)
    return df

def generate_rr(assets: pd.DataFrame, scenarios: pd.DataFrame) -> gr.Dataframe:
    # LIMIT INPUT SIZE - NEW
    assets = assets.head(MAX_ASSETS)
    scenarios = scenarios.head(MAX_SCENARIOS)
    
    # SIMPLIFIED PROMPT - NEW
    prompt = f"""
    Create risk register entries by connecting these assets to scenarios.
    Rules:
    1. Only connect relevant pairs
    2. Risk Score = Business Impact + Likelihood
    3. Output XML format shown below.
    
    Assets:
    {assets.to_xml(root_name='Assets', row_name='Asset', xml_declaration=False)}
    
    Scenarios:
    {scenarios.to_xml(root_name='Scenarios', row_name='Scenario', xml_declaration=False)}
    
    Required XML format:
    <result>
      <risks>
        <risk>
          <asset>Name</asset>
          <scenario>Name</scenario>
          <risk_score>Number</risk_score>
        </risk>
      </risks>
    </result>
    """

    try:
        response = client.chat.completions.create(
            model=LLM_MODEL_ID,
            messages=[
                {"role": "system", "content": "You are a risk assessment AI."},
                {"role": "user", "content": prompt}
            ],
            temperature=LLM_TEMPERATURE,
            max_tokens=LLM_MAX_OUT_TOKENS
        )

        xml_output = response.choices[0].message.content
        print("LLM Output:", xml_output)
        
        root = etree.fromstring(xml_output)
        risks = []
        
        for risk in root.xpath("//risk"):
            risks.append({
                "asset": risk.find("asset").text,
                "scenario": risk.find("scenario").text,
                "risk_score": int(risk.find("risk_score").text)
            })
            
        return pd.DataFrame(risks)
        
    except Exception as e:
        print(f"Error: {e}")
        return pd.DataFrame(columns=["asset", "scenario", "risk_score"])

# Gradio UI (unchanged)
with gr.Blocks() as app:
    with gr.Row():
        with gr.Accordion(open=False, label='Assets'):
            with gr.Row():
                rr_input_r_assets_f = gr.File(file_types=['.csv', '.xlsx', '.xls'])
            with gr.Row():
                rr_input_r_assets_load_btn = gr.Button('Load assets (CSV)')
            with gr.Row():
                rr_input_r_assets_inv = gr.Dataframe(label='Asset Inventory')
    with gr.Row():
        with gr.Accordion(open=False, label='Risk Scenarios'):
            with gr.Row():
                rr_input_r_scenarios_f = gr.File(file_types=['.csv', '.xlsx', '.xls'])
            with gr.Row():
                rr_input_r_scenarios_load_btn = gr.Button('Load risk scenarios (CSV)')
            with gr.Row():
                rr_input_r_scenarios_reg = gr.Dataframe(label='Risk Scenarios Register')
    with gr.Row():
        rr_create_btn = gr.Button('Create Risk Register')
    with gr.Row():
        rr_create_log = gr.Text(label='Creation log')
    with gr.Row():
        rr_output = gr.Dataframe(label='Generated Risk Register')

    rr_input_r_assets_load_btn.click(
        fn=load_csv_data, 
        inputs=[rr_input_r_assets_f], 
        outputs=[rr_input_r_assets_inv]
    )
    
    rr_input_r_scenarios_load_btn.click(
        fn=load_csv_data, 
        inputs=[rr_input_r_scenarios_f], 
        outputs=[rr_input_r_scenarios_reg]
    )
    
    rr_create_btn.click(
        fn=generate_rr, 
        inputs=[rr_input_r_assets_inv, rr_input_r_scenarios_reg], 
        outputs=[rr_output]
    )

app.launch(server_name='0.0.0.0', server_port=8080)