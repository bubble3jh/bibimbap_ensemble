import requests
import argparse
# parser = argparse.ArgumentParser(description='notification slack bot')
# parser.add_argument('--msg', type=str)
# args = parser.parse_args()

# json_data = {
#     'text': args.msg,
# }

# response = requests.post('https://hooks.slack.com/services/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', json=json_data)

def send_msg(msg1):
    json_data = {
        'text': msg1,
    }

    response = requests.post('https://hooks.slack.com/services/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', json=json_data)
