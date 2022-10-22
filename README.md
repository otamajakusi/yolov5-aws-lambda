# yolov5-aws-lambda

This repository is the sample for YOLOv5 on AWS lambda with OpenCV.

# Prerequisite

- AWS Account
- aws sam cli installed

# Build

```bash
sam build
```

# Run Local Server

```bash
sam local start-api
```

## Test

```bash
wget https://raw.githubusercontent.com/ultralytics/yolov5/master/data/images/zidane.jpg
image=$(base64 -w0 zidane.jpg)
echo { \"image\": \"${image}\" } | curl -X POST -H "Content-Type: application/json" -d @- http://127.0.0.1:3000/hello | jq -r .image | base64 -d > predicted.jpg
```

# Deploy to the AWS

```bash
sam deploly --guided
...
	Stack Name [sam-app]: yolov5-lambda-test --> (any stack name is ok)
	AWS Region [us-east-1]: --> (default or any region name you can use)
	#Shows you resources changes to be deployed and require a 'Y' to initiate deploy
	Confirm changes before deploy [y/N]: --> (N)
	#SAM needs permission to be able to create roles to connect to the resources in your template
	Allow SAM CLI IAM role creation [Y/n]: --> (Y)
	#Preserves the state of previously provisioned resources when an operation fails
	Disable rollback [y/N]: --> (N)
	HelloWorldFunction may not have authorization defined, Is this okay? [y/N]: --> (Y)
	Save arguments to configuration file [Y/n]: --> (Y)
	SAM configuration file [samconfig.toml]: --> (default)
	SAM configuration environment [default]: --> (default)
```

After saving deployment setting to `samconfig.toml`, you can delploy without `--guided` option.

```bash
sam deploy
```

You can see the endpoint at the end of the deployment.

## Test

You should replace endpoint below with the endpoint obtained on the deployment.

```bash
wget https://raw.githubusercontent.com/ultralytics/yolov5/master/data/images/zidane.jpg
image=$(base64 -w0 zidane.jpg)
echo { \"image\": \"${image}\" } | curl -X POST -H "Content-Type: application/json" -d @-  https://xxxxxxxxxx.execute-api.xxxxxxxx.amazonaws.com/Prod/hello/ | jq -r .image | base64 -d > predicted.jpg
```
