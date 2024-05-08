import anthropic
import os
import json

# Either we store the sysprompt in a text file and load it, or we use the sysprompt as the prefix.
# Maybe we have like, prefix[0:-2 = sysprompt, prefix[-1] = payload?
# Either way, it's not realistic to have the user write the sysprompt manually.

def API_CALL(sysprompt, payload, *args, keypath = "claude_api_key.txt", idx = payload, json_response = False):

    parent_folder = os.path.dirname(os.path.dirname(__name__))

    try:
        claude_key = open(parent_folder + keypath, "r").read()
    except FileNotFoundError:
        print("The file 'claude_api_key.txt' was not found in the expected location.")
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
    # However, if we're trying MIA extraction, it might be useful to only pass training data into the sysprompt
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