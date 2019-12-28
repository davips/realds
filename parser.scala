// def classif(l:String):String = {
// 	val arr = l.split(", ")
// 	return if (arr.last != "Infinity") (arr.dropRight(1) :+ "hit").mkString(", ") else l}

val lines = io.Source.fromFile("output.log").getLines.filter(!_.startsWith("]")).map(_.replace("[ \"", "").replace(", \"", "").replace("\"", "")).map(_ + "\n")

val atts = lines.take(1).toList(0).split(", ").dropRight(1).zipWithIndex.map{case (_, i) => s"@attribute a$i numeric"}.mkString("\n")

val fw = new java.io.FileWriter("output.arff")
fw.write("@relation tricoll\n")
fw.write(atts)
fw.write("\n@attribute time {\"Inf\", \"hit\"}\n")
fw.write("\n@data\n")
lines foreach fw.write
fw.close()
