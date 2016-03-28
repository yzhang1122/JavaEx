
public class MergeSort {

	public static void MergeSort(int[] array, int[] helper, int low, int high) {
		if (low < high) {
			int middle=(low + high) / 2;
			MergeSort(array, helper, low, middle);
			MergeSort(array, helper, middle+1, high);
			Merge(array, helper, low, middle, high);
		}
	}


	public static void Merge(int[] array, int[] helper, int low, int middle, int high) {
		for (int i=low; i<=high; i++) {
			helper[i] = array[i];
		}

		int hleft = low;
		int hright = middle+1;
		int current = low;
		while (hleft <= middle && hright <= high) {
			if (helper[hleft] < helper[hright]) {
				array[current] = helper[hleft];
				current++;
				hleft++;
			} else {
				array[current] = helper[hright];
				current++;
				hright++;
			}	
		}
		int remain = middle - hleft;
		for (int i=0; i<=remain; i++) {
			array[current+i] = helper[hleft+i];
		}
	}


	public static void main(String[] args){
		int[] array=new int[]{9,8,7,6,5,4,3,2,1,0};
		int[] helper=new int[array.length];


		for (int i=0; i<array.length; i++) {
			System.out.print(array[i]);
		}

		System.out.println();

		MergeSort(array, helper, 0, array.length-1);

		for (int i=0; i<array.length; i++) {
			System.out.print(array[i]);
		}

	}
}
