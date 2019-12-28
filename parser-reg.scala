val lines = io.Source.fromFile("output.txt").getLines.filter(!_.startsWith("]")).map(_.replace("[ \"", "").replace(", \"", "").replace("\"", "")).map(_ + "\n")

val atts = lines.take(1).toList(0).split(", ").dropRight(1).zipWithIndex.map{case (_, i) => s"@attribute a$i numeric"}.mkString("\n")

val fw = new java.io.FileWriter("reg-full.arff")
fw.write("@relation tricoll\n")
fw.write(atts)
fw.write("\n@attribute time numeric\n")
fw.write("\n@data\n")
lines foreach fw.write
fw.close()
