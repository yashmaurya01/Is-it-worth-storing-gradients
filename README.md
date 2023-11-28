Federated learning project:
-Gradient projection
-Adversarial attack detection
-Scaling based on contribution

Josh:
https://colab.research.google.com/drive/1h7X8BPx6EP-csB92BLYXb9JJLl-v1PnF#scrollTo=GsJXESEctevc
#old code for experiment
https://colab.research.google.com/drive/1Nug4qGqRgUn6k6UXhQAWZiQXYI1LzdmQ#scrollTo=8QaUV0L_OhXA
#AIJack: latest code

*push my code
*keep adding more code that can be used for Yash's end 


Anna:
https://colab.research.google.com/drive/1tbNp_TaqKp7uQs0L2hlCrS8qOHCiyNY5#scrollTo=dCk7GCi_ROOJ&uniqifier=1
-has done detection via feature. She will continue with her projectory calculation

Yash:
https://www.kaggle.com/yashmaurya/fl-project-experiments \
https://www.kaggle.com/code/yashmaurya/privacy-fl-project-experiment \

-Contribution repo working\
-Implement DP-SGD and how that affects performance while the attack is going on\
\
-Testing attacks on ShapleyFL\
-Using DP-SGD for local client privacy\
-Using DC-GAN for Private Synthetic Data Generation for privately sharing test dataset with server\
\
-Integrating Josh's code with ShapleyFL\
\
To add privacy to existing frameworks:
1) Use DPSGD (Fed Avg vs ShapleyFL) (Gradients safe)
2) Private synthetic test data (Fed Avg vs ShapleyFL) (Validation data shared to server is now secure) (Previously public dataset was required for validation on server end)
3) DPSGD + Private synthetic test data (Fed Avg vs ShapleyFL) (Everything private)
\
https://www.kaggle.com/code/yashmaurya/private-shapleyfl (Still need to test out dpsgd and setup privacy accounting)\
https://www.kaggle.com/code/yashmaurya/dp-dcgan-mnist (Need to tune it)
