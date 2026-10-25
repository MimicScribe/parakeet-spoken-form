import numscore as N
OK=[("62841-3-1","sixty two eight forty one dash three dash one"),
("192.168.246","one nine two one six eight two hundred forty six"),
("2133-1-26","twenty one thirty three one dash twenty six"),
("23,000","more than twenty three thousand residents"),
("15% 2019 7.3 7.4%","fifteen percent per year two thousand uh nineteen was uh over seven point three or seven point four percent"),
("50% 300 45 1200","fifty percent three hundred potency fourty five potency twelve hundred"),
("5.4%","Five points four percent"),
("1,500,000 1,800,000","between one million five hundred thousand and one million eight hundred thousand"),
("24 million 18 to 29 2004","twenty four million eighteen to twenty nine year olds from two thousand four"),
("1,400 1,660","about one thousand four hundred to one thousand six hundred sixty metres"),
("2020-2021","two thousand twenty two thousand twenty one draft"),
("2019 2020","twenty nineteen twenty twenty"),
("22.9 2209","twenty two dot nine twenty two oh nine"),
("M24KB","M two four KB"),("1966 1983","nineteen sixty six to nineteen eighty three"),
("0-521-64696-0","ISBN oh five two one six four six nine six oh"),
("0521646960","ISBN 0-521-64696-0"),("9am","nine AM"),("22nd","January twenty second"),
("10:30","ten thirty"),("9:00","at nine"),("1/2","one half"),("1.5","one and a half"),
("1980s","the nineteen eighties"),("767-9205","seven six seven nine two zero five"),
("555-1234","five five five one two three four"),("5577","double five seven seven"),
("2007","two thousand and seven"),("1905","nineteen oh five"),("150","one fifty"),
("0.03","zero point zero three"),("$1.5 million","one point five million dollars"),
("3D","three D"),("4A","four A"),("A-10","A ten"),("ESP8266","ESP eighty two sixty six"),
("100","a hundred"),("12","noon"),("9:30","half past nine"),("2024","twenty twenty four")]
SOFT=[("5.50","five fifty dollars","ok_point"), ("$5.50","five fifty","ok_point"),("9am","nineam","ok_glue"),("10S","TenS","ok_glue")]
BAD=[("528-2012","five hundred twenty eight two thousand twelfths"),
("20 41,500","twenty years four one thousand five hundred rupees"),
("9786404412345","nine trillion seven hundred eighty six billion"),
("1932 33","nineteen thirty two three"),("32-23","thirty two three"),("1988-89","nineteen eighty eighty nine"),
("2.1","Volume two texts"),("2.5","twenty five"),("3D","FreeD scan"),("4A","a fora"),("A-10","The A tan"),
("ESP8266","They called ESP826"),("978-1-44225930-2","nine seven eight one four four two two five nine three oh"),
("22","twenty two two"),("202","twenty two"),("2001 2002","two thousand and two thousand and two"),
("0.03","two point oh three cents"),("1975","nineteen seventy f seventy five"),("5 3","fifty three")]
f=0
for r,h in OK:
    v=N.verdict(r,h)
    if v!="ok": print("WANT ok ",v,"|",r,"|",h); f+=1
for r,h,w in SOFT:
    v=N.verdict(r,h)
    if v!=w: print("WANT",w,v,"|",r,"|",h); f+=1
for r,h in BAD:
    v=N.verdict(r,h)
    if v in N.OK_SET: print("WANT bad",v,"|",r,"|",h); f+=1
print("fail",f,"of",len(OK)+len(SOFT)+len(BAD))
for r,h,w,e in [("978-0-231-10483","ISBN 978-0-231-10483-0","ok_edge",True),("978-0-231-10483","ISBN 978-0-231-10483-0","bad",False),
 ("5 o'clock | 6 o'clock","at 5 o’clock, 6 o’clock","ok",False),("2% | 1","Typically 1 ?? 2% of","ok",False),("1st | 2nd","First, second","ok",False),
 ("6th | 2022","October 6th, uh, 2022.","ok",False),("0.1","Point 1","ok",False),("1800 to 7.616 | 1 billion | 2018","from 1 billion in 1800 to 7.616 billion in 2018","ok",False),
 ("4K","previous 4k","ok",False),("to a Newport place.","to a Newport place.","ok",False),("3D | 1996","The 1996 release of Quake one of the first 3D games","ok",False),
 ("10 | 15","between $10 and $15 an hour","ok",False),("97814","nine seven eight one four four","bad",False)]:
    v=N.verdict(r,h,e)
    if v!=w: print("WANT",w,v,"|",r,"|",h); f+=1
print("extra done, total fail",f)
for r,h,w in [("4K","previous four K","ok"),("250K","about two hundred fifty K","ok"),("4 o'clock","Four o'clock. Oh, so we","ok"),
 ("400th","The four hundredth anniversary","ok"),("136th | 116th","one hundred thirty sixth and one hundred sixteenth","ok"),
 ("1,100 | 1,400 | 3 | 4","between eleven hundred and fourteen hundred deaths three or four people","ok"),
 ("2,521 million","two thousand five hundred twenty one million tonnes","ok"),("3–5","Three to five million","ok"),
 ("0041 44212 4404","dial 0041442124404","ok"),("8 a.m. | 8 p.m.","on Wednesday AM and eight PM","bad"),("5 3","fifty three","bad")]:
    v=N.verdict(r,h)
    if (v in N.HEAD)!=(w=="ok"): print("WANT",w,v,"|",r,"|",h); f+=1
print("round3 total fail",f)
for r,h,w in [("0-451-12148-1","Signet ISBNO-451-12148-1","ok_glue"),("10,000–20,000","Some ten to twenty thousand men","ok"),("3–5","3 to 5 million","ok"),("4 o'clock","Four o'clock. Oh, so we","ok")]:
    v=N.verdict(r,h)
    if v!=w: print("WANT",w,v,"|",r,"|",h); f+=1
print("round4 total fail",f)
v=N.verdict("90377","ninety three hundred and seventy seven Sedna")
if v in N.HEAD: print("WANT bad",v); f+=1
print("round5 total fail",f)
for r,h in [("1999 to 2008","from nineteen ninety nine to two thousand and eight when"),("2014 to 2017","from twenty fourteen to two thousand seventeen."),("10,000–20,000","Some ten to twenty thousand men")]:
    v=N.verdict(r,h)
    if v not in N.HEAD: print("WANT ok",v,r,h); f+=1
print("round6 total fail",f)
print("--- v2.1")
for r,h,w in [("1918 9166 106,280 2213 273,039","one thousand nine hundred eighteen nine thousand one hundred sixty six one hundred six thousand two hundred eighty two thousand two hundred thirteen two hundred seventy three thousand thirty nine","ok_any"),
 ("2.5","two five","bad"),("3.14","three fourteen","bad"),("3.4","two points, three points, four points","bad"),
 ("2.5","two point five","ok"),("4.99","four ninety nine dollars","ok_point"),("5.50","five fifty","bad")]:
    v=N.verdict(r,h)
    good = v in N.HEAD if w=="ok_any" else (v not in N.OK_SET if w=="bad" else v==w)
    if not good: print("WANT",w,v,"|",r,"|",h); f+=1
print("v2.1 total fail",f)
