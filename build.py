import json
t=open("web/template.html").read()
open("web/index.html","w").write(t.replace("__MODEL__",open("web/model.json").read()))
print("built web/index.html")
