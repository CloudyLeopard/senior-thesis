from typing import Dict, List
import re
import json

request_mapping = {
    "entity": ["entity_name", "entity_category", "reasoning"],
    "info": ["info_description", "info_category", "reasoning", "entity_name"],
    "resc": ["func_name", "parameters", "purpose", "rank"]
}

def process_request(llm_response: str) -> List[Dict]:
    # TODO: there's probably a better way to do this
    # NOTE: maybe use pydantic with type validation??

    pattern = re.compile(r'(?<=\(|\|)([^|()]+)(?=\||\))')
    results = llm_response.splitlines()

    requests = [] # i call them "requests", like "info_requests" or "tool_requests"
    for result in results:
        match = pattern.findall(result)

        if not match: continue

        request_type = match[0]
        request = dict(zip(request_mapping[request_type], match[1:]))
        
        if request_type == "resc":
            # convert parameters from string to dict
            request["parameters"] = json.loads(request["parameters"])

        requests.append(request)

    return requests