import java.util.*;

public class Main {
    public static void main(String[] args) {
        int [] arr1 = { 4, 71, 8, 2, 12, 5, 23, 45, 20};
        int [] arr2 = {67, 3, 23, 20, 6, 31, 7, 9, 71};
        System.out.println(intersection(arr1, arr2));
    }

    public static List<Integer> intersection(int []arr1, int[] arr2) {
    	Arrays.sort(arr1);
    	Arrays.sort(arr2);
    	List<Integer> intersection = new ArrayList<Integer>();
    	int arr1Index = 0;
    	int arr2Index = 0;
    	// [2,4] [1,3,4] //2,3
    	while (arr1Index < arr1.length || arr2Index < arr2.length) {
    		if (arr1[arr1Index] == arr2[arr2Index]) {
    			intersection.add(arr1[arr1Index]);
    			arr1Index++;
    			arr2Index++;
    		} else if (arr1[arr1Index] < arr2[arr2Index]) {
    			arr1Index++;
    			if (arr1Index == arr1.length) {
    				break;
    			}
    		} else {
    			arr2Index++;
    			if (arr2Index == arr2.length) {
    				break;
    			}
    		}
    	}
    	
    	return intersection;
	}
}
