import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.*;

public class Morse {
    private static String filePath = "/Users/yzhang/JavaEx/src/problemc.in.txt";

    public static void main(String[] args) throws IOException {
        BufferedReader br = new BufferedReader(new FileReader(filePath));
        String line = null;
        int count = 0;
        Map<String, String> map = new HashMap<String, String>();
        Set<String> set= new HashSet<String>();
        List<String> list = new LinkedList<String>();
        Map<String, String> dictionary = new HashMap<String, String>();


        /*
        *
        * converting file to data structure
        *
        * */
        while ((line = br.readLine()) != null) {
            if (line.equals("*")) {
                count++;
                continue;
            }
            if (count == 0) {
                String[] strs = line.split(" ");
                map.put(strs[0], strs[1]);
            } else if (count == 1) {
                set.add(line);
            } else if (count == 2 && !line.equals("*")) {
                String[] strs = line.split(" ");
                for (String str : strs) {
                    list.add(str);
                }
            }
        }


        /*
        *
        * creating a dic
        * */
        for (String word : set) {
            char[] arr = word.toCharArray();
            StringBuilder sb = new StringBuilder();

            for (Character c : arr) {
                sb.append(map.get(String.valueOf(c)));
            }
            dictionary.put(word, sb.toString());
        }


        /*
        * printing out results
        * */
        for (String str : list) {
            List<String> matches = new LinkedList<String>();
            List<String> similarList = new LinkedList<String>();
            for (String key : dictionary.keySet()) {
                String value = dictionary.get(key);
                if (value.equals(str)) {
                    matches.add(key);

                } else if (value.contains(str)){
                    similarList.add(value);
                }
            }
            if (matches.size() == 1) {
                System.out.println(matches.get(0));
            } else if (matches.size() > 1) {
                System.out.println(matches.get(0) + "!");
            } else if (matches.size() == 0 && similarList.size() > 0) {
                String temp = similarList.get(0);
                int maxLen = similarList.get(0).length();
                for (int i=1; i<similarList.size(); i++) {
                    int len = similarList.get(i).length();
                    if (len > maxLen) {
                        maxLen = len;
                        temp = similarList.get(i);
                    }
                }
                for (String key : dictionary.keySet()) {
                    if (dictionary.get(key).equals(temp)) {
                        System.out.println(key + "?");
                    }
                }
            }
        }
    }
}
