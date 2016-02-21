package algorithms;


import java.util.LinkedList;
import java.util.List;

public class SummaryRanges {
    public List<String> summaryRanges(int[] nums) {
        List<String> list = new LinkedList<String>();
        if(nums==null || nums.length < 1) {
            return  list;
        }
        int start=0;
        int end=0;

        while (end < nums.length) {
            if (end+1 < nums.length && nums[end+1]==nums[end]+1) {
                end++;
            } else {
                if (start == end) {
                    list.add(Integer.toString(nums[start]));
                } else {
                    String str = nums[start] + "->" + nums[end];
                    list.add(str);
                }
                end++;
                start = end;
            }
        }
        return list;
    }
}



