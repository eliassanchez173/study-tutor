FROM public.ecr.aws/lambda/python:3.12

WORKDIR /var/task

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py security.py lambda_handler.py ./

# CMD is overridden per-function in template.yaml
CMD ["lambda_handler.chat"]