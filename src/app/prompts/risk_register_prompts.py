import os
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

import gradio as gr
import pandas as pd
from openai import OpenAI
from lxml import etree
from prompts import rr_prompt as prompt

# Initialize OpenAI client
client = OpenAI()

# Configuration
LLM_TEMPERATURE = 0.0
LLM_MODEL_ID = 'gpt-4'
LLM_MAX_OUT_TOKENS = 4000  # Increased to accommodate solutions
MAX_ASSETS = 5
MAX_SCENARIOS = 5

def load_csv_data(input_file: str) -> gr.Dataframe:
    input_file = input_file.name
    print(f'Opening CSV: {input_file}')
    df = pd.read_csv(input_file)
    return df

def generate_rr(assets: pd.DataFrame, scenarios: pd.DataFrame) -> gr.Dataframe:
    # Limit input size
    assets = assets.head(MAX_ASSETS)
    scenarios = scenarios.head(MAX_SCENARIOS)
    
    rules = """
    1. Only output requested schema with solutions
    2. For each risk, provide one technical and one administrative solution
    3. Solutions should be practical and cost-effective
    """

    # Prepare inputs for the prompt
    chain_input = {
        'rules': rules,
        'assets': assets.to_xml(root_name='Assets', row_name='Asset', xml_declaration=False),
        'scenarios': scenarios.to_xml(root_name='Scenarios', row_name='Scenario', xml_declaration=False)
    }

    try:
        # Call OpenAI API
        response = client.chat.completions.create(
            model=LLM_MODEL_ID,
            messages=[
                {"role": "system", "content": prompt.format_messages(**chain_input)[0].content},
                {"role": "user", "content": prompt.format_messages(**chain_input)[1].content}
            ],
            temperature=LLM_TEMPERATURE,
            max_tokens=LLM_MAX_OUT_TOKENS
        )

        xml_output = response.choices[0].message.content
        print("LLM Output:", xml_output)
        
        # Parse XML output
        root = etree.fromstring(xml_output)
        risks = []
        
        for risk in root.xpath("//risk"):
            risk_data = {
                "asset": risk.find("asset").text,
                "scenario": risk.find("scenario").text,
                "risk_score": int(risk.find("risk_score").text),
                "solution": risk.find("solution").text if risk.find("solution") is not None else "No solution provided"
            }
            risks.append(risk_data)
            
        return pd.DataFrame(risks)
        
    except Exception as e:
        print(f"Error: {e}")
        return pd.DataFrame(columns=["asset", "scenario", "risk_score", "solution"])

# Gradio UI with solutions display
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
        rr_output = gr.Dataframe(
            label='Generated Risk Register',
            headers=["Asset", "Scenario", "Risk Score", "Recommended Solutions"],
            datatype=["str", "str", "number", "str"]
        )
    with gr.Row():
        with gr.Accordion("Detailed Solutions", open=False):
            rr_solutions = gr.DataFrame(
                label="Mitigation Strategies",
                headers=["Asset", "Solutions"],
                datatype=["str", "str"]
            )

    # UI Logic
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
    
    def update_ui(assets, scenarios):
        df = generate_rr(assets, scenarios)
        solutions_df = df[["asset", "solution"]]
        return df, solutions_df
    
    rr_create_btn.click(
        fn=update_ui, 
        inputs=[rr_input_r_assets_inv, rr_input_r_scenarios_reg], 
        outputs=[rr_output, rr_solutions]
    )

app.launch(server_name='0.0.0.0', server_port=8080)