from langchain_core.prompts import ChatPromptTemplate

example_1 = """
Human:

Please connect the following assets:
<Assets>
  <Asset>
    <index>0</index>
    <asset_name>Web Server Infrastructure</asset_name>
    <business_impact>1</business_impact>
    <description>Hosts the company’s web application, serving as the primary point of interaction with users. Critical for service availability and directly impacts revenue and customer trust.</description>
  </Asset>
  <Asset>
    <index>1</index>
    <asset_name>Customer Database</asset_name>
    <business_impact>2</business_impact>
    <description>Contains sensitive customer information including personal data, payment details, and transaction history. Its integrity, confidentiality, and availability are paramount for compliance, trust, and operational continuity.</description>
  </Asset>
  <Asset>
    <index>2</index>
    <asset_name>Logging and Monitoring Systems</asset_name>
    <business_impact>2</business_impact>
    <description>Collects and analyzes logs from various systems for monitoring performance, detecting anomalies, and facilitating incident response. Critical for operational awareness and security.</description>
  </Asset>
  <Asset>
    <index>3</index>
    <asset_name>Development and Testing Environments</asset_name>
    <business_impact>1</business_impact>
    <description>Used by the development team for building and testing new features and updates before they are deployed to the production environment. Important for maintaining the pace of innovation while ensuring application stability and security.</description>
  </Asset>
</Assets>
to the following scenarios:
<Scenarios>
  <Scenario>
    <index>0</index>
    <risk_scenario_name>DDoS Attack on Service Infrastructure</risk_scenario_name>
    <likelihood>2</likelihood>
    <description>An attacker targets the company's service infrastructure with a Distributed Denial of Service (DDoS) attack, overwhelming the servers with traffic and making the service unavailable to legitimate users.</description>
  </Scenario>
  <Scenario>
    <index>1</index>
    <risk_scenario_name>Data Breach Through Phishing Attack</risk_scenario_name>
    <likelihood>3</likelihood>
    <description>An attacker successfully deceives an employee into revealing their credentials through a phishing email. The attacker gains unauthorized access to sensitive data stored on the company’s network.</description>
  </Scenario>
  <Scenario>
    <index>2</index>
    <risk_scenario_name>Ransomware Infection</risk_scenario_name>
    <likelihood>2</likelihood>
    <description>Malicious software encrypts critical data and systems, rendering them unusable. The attacker demands a ransom payment for the decryption key. This scenario can disrupt operations and lead to data loss if backups are also compromised.</description>
  </Scenario>
  <Scenario>
    <index>3</index>
    <risk_scenario_name>Third-Party Service Failure</risk_scenario_name>
    <likelihood>3</likelihood>
    <description>A critical third-party service provider experiences a failure or security breach, impacting the company’s ability to offer its digital service, either through direct service disruption or through a breach of data shared with the third party.</description>
  </Scenario>
</Scenarios>
Please output the result in requested schema.

Assistant:

<result>
  <thinking>
    <step num="1">
      Web Server Infrastructure - DDoS Attack on Service Infrastructure (relevant because a DDoS attack directly targets and impacts web server infrastructure)
      Customer Database - Data Breach Through Phishing Attack (relevant because a phishing attack could expose credentials and enable unauthorized access to customer data) 
      Customer Database - Ransomware Infection (relevant because ransomware could encrypt and disrupt access to customer data)
      Logging and Monitoring Systems - Third-Party Service Failure (relevant because monitoring systems are needed to detect failures in third-party services)
    </step>
    <step num="2">
      Web Server Infrastructure - DDoS Attack on Service Infrastructure: 3 (1+2)
      Customer Database - Data Breach Through Phishing Attack: 4 (1+3)
      Customer Database - Ransomware Infection: 3 (1+2)
      Logging and Monitoring Systems - Third-Party Service Failure: 5 (2+3)
    </step>
  </thinking>
<risks>
  <risk num="1">
    <asset>Web Server Infrastructure</asset> 
    <scenario>DDoS Attack on Service Infrastructure</scenario>
    <risk_score>5</risk_score>
  </risk>  
  <risk num="2">
    <asset>Customer Database</asset>
    <scenario>Data Breach Through Phishing Attack</scenario>
    <risk_score>6</risk_score>
  </risk>
  <risk num="3">
    <asset>Customer Database</asset>
    <scenario>Ransomware Infection</scenario>
    <risk_score>5</risk_score>
  </risk>  
  <risk num="4">
    <asset>Logging and Monitoring Systems</asset>
    <scenario>Third-Party Service Failure</scenario>
    <risk_score>5</risk_score>
  </risk>
</risks>  
</result>
    """

system_str_f = """
You are an IT risk management expert. You are being asked to perform various 
tasks related to  organizational IT security risk management. 

You are brief and to the point. You are good at following instructions and 
providing clear and concise information. 

Here are some additional important rules for you:
<rules>
{rules}
</rules>

The current task is about creating a risk register. A risk register is a
document that contains information about the risks that an organization faces. 
It consists of a list of risks and their associated metadata. 

Each risk is constructed based on the connection between an asset and a scenario.
The risk score is calculated for each connection based on the formula:
Risk = Probability + Impact

Your task is to only connect the assets to the scenarios that together form 
a meaningful risk and avoid connecting assets to scenarios that are not relevant.

For example, combination of an asset "Server" and a scenario "Server Downtime" 
is meaningful, but combination of an asset "Backup" and a scenario "Server Downtime"
does not make sense.

Your second task is to calculate the risk score for each connection that you create.

Below are examples of the task execution:
<examples>
  <example num="1">
{example_1}
  </example>
</examples>

Please execute and document the following thinking steps:
<thinking_steps>
  <step num="1">
    For each asset, connect it to the relevant scenarios. Write down the selected
    connections with the short explanation of why the connection is meaningful.
  </step>
  <step num="2">
    For each connection, calculate the risk score based on the requested formula.
  </step>
</thinking_steps>

Please output the result in the following schema:
```
<result>
<thinking>
  <step num="1">
    Your notes for step 1
  </step>
  <step num="2">
    Your notes for step 2
  </step>
</thinking>
<risks>
  <risk num="1">
    <asset>Sample asset 1</asset>
    <scenario>Sample scenario 1</scenario>
    <risk_score>3</risk_score>
  </risk>
    <asset>Sample asset 2</asset>
    <scenario>Sample scenario 2</scenario>
    <risk_score>4</risk_score>
  <risk num="2">
  </risk>
    <asset>Sample asset 3</asset>
    <scenario>Sample scenario 3</scenario>
    <risk_score>1</risk_score>
  <risk num="3">
  </risk>
  ... // more risks
</risks>
</result>
```
    """

system_str = system_str_f.format(
    rules='{rules}', 
    example_1=example_1)

human_str = """
Please connect the following assets:
{assets}
to the following scenarios:
{scenarios}
Please output the result in requested schema.
    """
  
rr_prompt = ChatPromptTemplate.from_messages([
    ("system", system_str),
    ("human", human_str)
])

