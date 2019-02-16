insert(py.sys.path,int32(0),'');
flag = int32(bitor(2, 8));
py.sys.setdlopenflags(flag);

py.syngen_matlab.launch()
py.syngen_matlab.insert("2")
py.syngen_matlab.insert("9")
py.syngen_matlab.insert("null")

py.syngen_matlab.get()
py.syngen_matlab.get()
py.syngen_matlab.get()

py.syngen_matlab.kill()
