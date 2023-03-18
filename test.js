var searchInsert = function (nums, target) {
    let first = 0;
    let last = nums.length - 1
    let middle = parseInt((last + first) / 2)

    while (first < last) {
        console.log("before",first, middle, last)
        if (target > nums[middle])
            first = middle
        else if (target < nums[middle])
            last = middle
        else{
            return middle;
        }
        middle = parseInt((last + first) / 2)
    }
    return first;
}
console.log(searchInsert([1,3,5,6],2))