# convert ref.txt and tst.txt into ref.xml and tst.xml
import os 

def make_html_safe(s):
  """Replace any angled brackets in string s to avoid interfering with HTML attention visualizer."""
  s = s.replace("&", "&amp;")
  s = s.replace("<", "&lt;")
  s = s.replace(">", "&gt;")
  #s = s.replace("\"", "&quot;")
  #s = s.replace("\'", "&apos;") 
  return s

dir = "."
with open(os.path.join(dir, "../ref.txt"),'r') as f:
  ref = f.readlines()
with open(os.path.join(dir, "../tst.txt"),'r') as f:
  tst = f.readlines()

idx = 100

# write to ref.xml
with open(os.path.join(dir, "example/ref.xml"),'r') as f:
  lines = f.readlines()    
f = open(os.path.join(dir, "ref.xml"),'w')
f.write("".join(lines[:6]))
for i,l in enumerate(ref[:idx]):
  l = "<seg id=\"" + str(i+1) + "\"> " + make_html_safe(l.strip()) + " </seg>"
  f.write(l + "\n")
f.write("".join(lines[7:]))
f.close()

# write to src.xml
with open(os.path.join(dir, "example/src.xml"),'r') as f:
  lines = f.readlines()    
f = open(os.path.join(dir, "src.xml"),'w')
f.write("".join(lines[:6]))
for i,l in enumerate(ref[:idx]):
  l = "<seg id=\"" + str(i+1) + "\"> " + make_html_safe(l.strip()) + " </seg>"
  f.write(l + "\n")
f.write("".join(lines[7:]))
f.close()

# write to tst.xml
with open(os.path.join(dir, "example/tst.xml"),'r') as f:
  lines = f.readlines()    
f = open(os.path.join(dir, "tst.xml"),'w')
f.write("".join(lines[:6]))
for i,l in enumerate(tst[:idx]):
  l = "<seg id=\"" + str(i+1) + "\"> " + make_html_safe(l.strip()) + " </seg>"
  f.write(l + "\n")
f.write("".join(lines[7:]))
f.close()

