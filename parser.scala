def classif(l:String):String = {
	val arr = l.split(", ")
	return if (arr.last != "Infinity") (arr.dropRight(1) :+ "hit").mkString(", ") else l}

val fw = new java.io.FileWriter("output-parsed.txt")
val lines = io.Source.fromFile("output.txt").getLines.filter(!_.startsWith("]")).map(_.replace("[ \"", "").replace(", \"", "").replace("\"", "")).map(classif).map(_ + "\n")
lines foreach fw.write
fw.close()
