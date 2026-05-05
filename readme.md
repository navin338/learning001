### Execution
```
1 workflow pipeline have the yaml code for local hosted & the githb hosted agent to run the task
2 here we cannot directly use the local host URL to access the kubeflow from the github hosted agent so we are using "ngrok" to define a any DNS name & https request to get another URL
example execute this line
 
 ngrok http 30008

 this should run on the back end
 gives the URL like this https://filling-risk-lingo.ngrok-free.dev/

3 WITH DOCKER FILE: once you push the code it will run automatically, use this URL to access the swagger to check

http://localhost:30009/docs#/default/predict_predict_post

4 As well for usin the local host run this cmd to this specific path 

D:\selfhosted_runner\actions-runner> ./run.cmd





```

```

```