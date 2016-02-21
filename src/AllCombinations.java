package algorithms;

import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;


public class AllCombinations {
    public Map<Integer, String[]> map = new HashMap<Integer, String[]>();

    public void setMap() {
        map.put(1, new String[] {"1"});
        map.put(2, new String[] {"2", "a", "b", "c"});
        map.put(3, new String[] {"3", "d", "e", "f"});
        map.put(4, new String[] {"4", "g", "h", "i"});
        map.put(5, new String[] {"5", "j", "k", "l"});
        map.put(6, new String[] {"6", "m", "n", "o"});
        map.put(7, new String[] {"7", "p", "q", "r", "s"});
        map.put(8, new String[] {"8", "t", "u", "v"});
        map.put(9, new String[] {"9", "w", "x", "y", "z"});

    }

    public void getAllCombinations(int[] input, int level, LinkedList<String> result, LinkedList<LinkedList<String>> results) {
        if (level >= input.length) {
            results.add(result);
        } else {
            for (String s : map.get(input[level])) {
                LinkedList<String> tmp = new LinkedList<String>(result);
                tmp.add(s);
                getAllCombinations(input, level+1, tmp, results);
            }
        }
    }
}
