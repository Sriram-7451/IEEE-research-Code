import os
import re
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

import gradio as gr
import pandas as pd
from openai import OpenAI
from lxml import etree
from lxml.etree import XMLSyntaxError
from io import BytesIO

# Initialize OpenAI client
client = OpenAI()

# Configuration
LLM_TEMPERATURE = 0.0
LLM_MODEL_ID = 'gpt-4'
LLM_MAX_OUT_TOKENS = 4000
MAX_ASSETS = 5
MAX_SCENARIOS = 5

def load_csv_data(input_file: str) -> gr.Dataframe:
    input_file = input_file.name
    print(f'Opening CSV: {input_file}')
    df = pd.read_csv(input_file)
    return df

def generate_rr(assets: pd.DataFrame, scenarios: pd.DataFrame) -> gr.Dataframe:
    assets = assets.head(MAX_ASSETS)
    scenarios = scenarios.head(MAX_SCENARIOS)
    
    prompt = f"""
    Create risk register entries with mitigation suggestions using:
    
    Assets: {assets.to_xml(root_name='Assets', row_name='Asset')}
    Scenarios: {scenarios.to_xml(root_name='Scenarios', row_name='Scenario')}
    
    Required XML format:
    <RiskRegister>
      <Risk>
        <Asset>Name</Asset>
        <Scenario>Name</Scenario>
        <RiskScore>1-10</RiskScore>
        <SuggestedSolution>Your mitigation strategy here</SuggestedSolution>
      </Risk>
    </RiskRegister>
    """

    try:
        response = client.chat.completions.create(
            model=LLM_MODEL_ID,
            messages=[
                {"role": "system", "content": "You are a risk assessment AI that outputs perfect XML without XML declaration."},
                {"role": "user", "content": prompt}
            ],
            temperature=LLM_TEMPERATURE,
            max_tokens=LLM_MAX_OUT_TOKENS
        )

        xml_output = response.choices[0].message.content
        print("Raw LLM Output:", xml_output)
        
        # Clean XML output
        xml_output = xml_output.strip()
        
        # Remove XML declaration if present
        xml_output = re.sub(r'^<\?xml.*?\?>', '', xml_output).strip()
        
        # Remove Markdown code blocks if present
        if "```xml" in xml_output:
            xml_output = xml_output.split("```xml")[1].split("```")[0].strip()
        
        # Ensure proper closing
        if not xml_output.endswith(">"):
            xml_output += ">"
        
        # Parse XML as bytes to avoid encoding declaration issues
        root = etree.fromstring(xml_output.encode('utf-8'))
        risks = []
        
        for risk in root.xpath("//Risk"):
            risks.append({
                "Asset": risk.find("Asset").text,
                "Scenario": risk.find("Scenario").text,
                "Risk Score": int(risk.find("RiskScore").text),
                "Suggested Solution": risk.find("SuggestedSolution").text if risk.find("SuggestedSolution") is not None else "No suggestion provided"
            })
            
        result_df = pd.DataFrame(risks)
        print("\nGenerated DataFrame Columns:", result_df.columns.tolist())
        print("DataFrame Preview:\n", result_df)
        
        return result_df
        
    except Exception as e:
        print(f"Error: {type(e).__name__}: {e}\nLLM Output: {xml_output}")
        return pd.DataFrame(columns=["Asset", "Scenario", "Risk Score", "Suggested Solution"])

# Gradio UI with explicit column configuration
with gr.Blocks() as app:
    with gr.Row():
        with gr.Accordion(open=False, label='Assets'):
            rr_input_r_assets_f = gr.File(file_types=['.csv'])
            rr_input_r_assets_load_btn = gr.Button('Load Assets')
            rr_input_r_assets_inv = gr.Dataframe(
                label='Asset Inventory',
                headers=["Asset", "Description", "Criticality"],
                datatype=["str", "str", "str"]
            )
    
    with gr.Row():
        with gr.Accordion(open=False, label='Risk Scenarios'):
            rr_input_r_scenarios_f = gr.File(file_types=['.csv'])
            rr_input_r_scenarios_load_btn = gr.Button('Load Scenarios')
            rr_input_r_scenarios_reg = gr.Dataframe(
                label='Scenario Register',
                headers=["Scenario", "Likelihood", "Impact"],
                datatype=["str", "number", "number"]
            )
    
    rr_create_btn = gr.Button('Generate Risk Register', variant='primary')
    rr_create_log = gr.Textbox(label='Processing Log', interactive=False)
    
    # Configure output dataframe - removed max_rows parameter
    rr_output = gr.Dataframe(
        label='Risk Register Output',
        headers=["Asset", "Scenario", "Risk Score", "Suggested Solution"],
        datatype=["str", "str", "number", "str"],
        interactive=True
    )

    # Event handlers
    rr_input_r_assets_load_btn.click(
        load_csv_data,
        inputs=rr_input_r_assets_f,
        outputs=rr_input_r_assets_inv
    )
    
    rr_input_r_scenarios_load_btn.click(
        load_csv_data,
        inputs=rr_input_r_scenarios_f,
        outputs=rr_input_r_scenarios_reg
    )
    
    rr_create_btn.click(
        generate_rr,
        inputs=[rr_input_r_assets_inv, rr_input_r_scenarios_reg],
        outputs=rr_output
    )

app.launch(server_name='0.0.0.0', server_port=8080)