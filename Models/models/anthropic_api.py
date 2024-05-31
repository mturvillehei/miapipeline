import anthropic
import os
import json

def OPUS_API_CALL(payload, *args, syspromptpath = "claude_opus_sysprompt.txt", keypath = "claude_api_key.txt", idx = None, json_response = False):
        
    ''' 

    Parameters:
        payload: the text to prompt
        *args: nothing yet
        syspromptpath: location of the sysprompt to use

    Returns:

        output: Each row is a dict containing the output tokens of the model.

    ''' 
    parent_folder = os.path.dirname(os.path.dirname(__name__))
    print(f"Searching for the API key and sysprompt in {parent_folder}")
    try:
        claude_key = open(parent_folder + keypath, "r").read()
    except FileNotFoundError:
        print(f"The file {keypath} was not found in the expected location.")
        print("Please check the file path and ensure the file exists.")

    try:
        sysprompt = open(parent_folder + syspromptpath, "r").read()
    except FileNotFoundError:
        print(f"The file {syspromptpath} was not found in the expected location.")
        print("Please check the file path and ensure the file exists.")
    
    client = anthropic.Anthropic(
        api_key=claude_key,
        max_retries=2,  
        timeout=60.0    
    )

    try:
        message = client.messages.create(
            model="claude-3-opus-20240229",
            max_tokens=1000,
            temperature=0,
            system=sysprompt,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": payload
                        }
                    ]
                }
            ]
        )

    except anthropic.APIConnectionError as e:
        print(f"The server could not be reached for prompt {idx}.jpg")
        print(f"Underlying exception: {e.__cause__}")
        
    except anthropic.RateLimitError as e:
        print(f"A 429 status code was received for prompt {idx}.jpg; backing off")
        
    except anthropic.APIStatusError as e:
        print(f"A non-200 status code was received for prompt {idx}.jpg")
        print(f"Status code: {e.status_code}")
        print(f"Response: {e.response}")
    
    # Response can be formatted as a JSON output. This can be set in the sysprompt.
    if json_response:
        try:
            OUTPUT = json.loads(message.content[0].text)
            
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON response for image {idx}.jpg: {message.content}")
            
        except:
            print(f"Some error with {idx}, giving output {message.content[0].text}. Skipping")
    else:
        OUTPUT = message.content[0].text

    return OUTPUT