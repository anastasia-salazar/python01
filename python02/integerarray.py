
nums = [1, 2, 3, 4, 5, 6, 7, 8, 8, 10]

class Solution(object):
    def containsDuplicate(self, nums):
        """
        :type nums: List[int]
        :rtype: bool
        """
        x = len(nums)
        print(f"Array Length: {x}")

        for i in range(x):
            for j in range(x):
                if i != j and nums[i] == nums[j]:
                    return True

        return False

    def primeFac(self, n):
        """
        :type n: int
        :rtype: List[int]
        """
        factors = []
        for i in range(2, n + 1):
            while n % i == 0:
                factors.append(i)
                n //= i
        return factors

    def fizzbuzz(self, n):
        """
        1 to n
        multiples of 3 print Fizz
        multiples of 5 print Buzz
        both multiples of 3 and 5 print FizzBuzz
        """
        result = []
        for i in range(1, n + 1):
            if i % 3 == 0 and i % 5 == 0:
                result.append("FizzBuzz")
            elif i % 3 == 0:
                result.append("Fizz")
            elif i % 5 == 0:
                result.append("Buzz")
            else:
                result.append(str(i))
        return result

    def maxArea(self, height):
        left, right = 0, len(height) - 1
        max_area = 0

        while left < right:
            area = min(height[left], height[right]) * (right - left)
            max_area = max(max_area, area)

            if height[left] <= height[right]:
                left += 1
            else:
                right -= 1

        return max_area

    def palindromeFinder(self, s):
        """
        :type s: str
        :rtype: bool
        """
        cleaned = "".join(ch.lower() for ch in s if ch.isalnum())
        return cleaned == cleaned[::-1]

    def longestCommonPrefix(self, strs):
        """
        :type strs: List[str]
        :rtype: str
        """
        if not strs:
            return ""

        prefix = strs[0]
        for word in strs[1:]:
            while not word.startswith(prefix):
                prefix = prefix[:-1]
                if not prefix:
                    return ""

        return prefix

    def camelCase(self, s):
        """
        :type s: str
        :rtype: str
        """
        words = s.split()
        if not words:
            return ""

        return words[0].lower() + "".join(word.capitalize() for word in words[1:])

    def longestPalindrome(self, s):
        """
        :type s: str
        :rtype: str
        """
        if not s:
            return ""

        start = 0
        max_len = 1

        def expand(left, right):
            nonlocal start, max_len
            while left >= 0 and right < len(s) and s[left] == s[right]:
                curr_len = right - left + 1
                if curr_len > max_len:
                    start = left
                    max_len = curr_len
                left -= 1
                right += 1

        for i in range(len(s)):
            expand(i, i)
            expand(i, i + 1)

        return s[start:start + max_len]

    def groupAnagrams(self, strs):
        """
        :type strs: List[str]
        :rtype: List[List[str]]
        """
        groups = {}
        for word in strs:
            key = ''.join(sorted(word))
            if key not in groups:
                groups[key] = []
            groups[key].append(word)
        return list(groups.values())

    def twoSum(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[int]
        """
        seen = {}
        for i, value in enumerate(nums):
            complement = target - value
            if complement in seen:
                return [seen[complement], i]
            seen[value] = i
        return []

    def topKFrequent(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: List[int]
        """
        counts = {}
        for num in nums:
            counts[num] = counts.get(num, 0) + 1

        return [num for num, _ in sorted(counts.items(), key=lambda item: item[1], reverse=True)[:k]]

    def commonElements(self, arr1, arr2):
        """
        :type arr1: List[int]
        :type arr2: List[int]
        :rtype: List[int]
        """
        set2 = set(arr2)
        return sorted({x for x in arr1 if x in set2})

    def romanToInt(self, s):
        """
        :type s: str
        :rtype: int
        """
        values = {
            'I': 1,
            'V': 5,
            'X': 10,
            'L': 50,
            'C': 100,
            'D': 500,
            'M': 1000
        }

        total = 0
        for i in range(len(s)):
            if i + 1 < len(s) and values[s[i]] < values[s[i + 1]]:
                total -= values[s[i]]
            else:
                total += values[s[i]]

        return total


class MinStack(object):
    def __init__(self):
        self.stack = []
        self.min_stack = []

    def push(self, val):
        self.stack.append(val)
        if not self.min_stack or val <= self.min_stack[-1]:
            self.min_stack.append(val)

    def pop(self):
        if not self.stack:
            return None
        val = self.stack.pop()
        if val == self.min_stack[-1]:
            self.min_stack.pop()
        return val

    def top(self):
        if not self.stack:
            return None
        return self.stack[-1]

    def getMin(self):
        if not self.min_stack:
            return None
        return self.min_stack[-1]


if __name__ == "__main__":
    solution = Solution()
    result = solution.containsDuplicate(nums)
    print(f"Contains Duplicate: {result}")

    number = 60
    factors = solution.primeFac(number)
    print(f"Prime Factors of {number}: {factors}")

    n = 45
    fb_result = solution.fizzbuzz(n)
    print(f"FizzBuzz for {n}: {fb_result}")

    heights = [1, 8, 6, 2, 5, 4, 8, 3, 7]
    area = solution.maxArea(heights)
    print(f"Max Area: {area}")

    phrase = "A man, a plan, a canal: Panama"
    print(f"Palindrome check for '{phrase}': {solution.palindromeFinder(phrase)}")

    words = ["geeksforgeeks", "geeks", "geek", "geezer"]
    print(f"Longest common prefix: {solution.longestCommonPrefix(words)}")

    sentence = "hello world from python"
    print(f"CamelCase: {solution.camelCase(sentence)}")

    text = "babad"
    print(f"Longest palindromic substring: {solution.longestPalindrome(text)}")

    anagrams = ["eat", "tea", "tan", "ate", "nat", "bat"]
    print(f"Anagram groups: {solution.groupAnagrams(anagrams)}")

    two_sum_nums = [2, 7, 11, 15]
    two_sum_target = 9
    print(f"Two-sum indices: {solution.twoSum(two_sum_nums, two_sum_target)}")

    freq_nums = [1, 1, 1, 2, 2, 3]
    print(f"Top k frequent: {solution.topKFrequent(freq_nums, 2)}")

    common_a = [1, 2, 3, 4, 5]
    common_b = [3, 5, 7, 9]
    print(f"Common elements: {solution.commonElements(common_a, common_b)}")

    roman = "LVIII"
    print(f"Roman numeral {roman} = {solution.romanToInt(roman)}")

    stack = MinStack()
    for value in [5, 2, 7, 1, 3]:
        stack.push(value)
        print(f"Push {value}, min={stack.getMin()}")

    print(f"Top={stack.top()}, min={stack.getMin()}")
    stack.pop()
    print(f"After pop, top={stack.top()}, min={stack.getMin()}")