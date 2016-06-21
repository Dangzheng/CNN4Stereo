## 2. Scoping（作用域）
### 2.1 Namespaces（命名空间）
```
鼓励在 .cc 文件内使用匿名名字空间。使用具名的名字空间时，其名称可基于项目名或相对路径。
禁止使用 using 指示（using-directive）。
禁止使用内联命名空间（inline namespace）。
```

**定义**：
名字空间将全局作用域细分为独立的，具名的作用域，可有效防止全局作用域的命名冲突。

**优点**：
虽然类已经提供了（可嵌套的）命名轴线 (注:将命名分割在不同类的作用域内)，命名空间在这基础上又封装了一层。

举个例子，两个不同项目的全局作用域都有一个类 `foo` ,在编译或者运行时会有冲突。如果每个项目将代码置于不同的命名空间之中，比如 `project::Foo` 和 `project2::Foo` ，现在他们作为不同的符号自然不会冲突。

内联命名空间会自动把内部的标识符放到外层作用域，比如：

```
namespace X {
inline namespace Y {
void foo();
}
}
```

`X::Y::foo()` 与 `X::foo()` 彼此可代替。内联命名空间主要用来保持跨版本的 ABI 兼容性。

**缺点**：

命名空间具有迷惑性，因为他们和类一样提供了额外的（可嵌套的）命名轴线。

命名空间很容易令人迷惑，毕竟它们不再受其声明所在命名空间的限制。内联命名空间只在大型版本控制里有用。

在头文件中使用匿名空间导致违背 C++ 的唯一定义原则 (One Definition Rule (ODR))。

**结论**：
根据下文将要提到的策略合理使用命名空间。

### 2.1.1 Unnamed Namespaces (匿名命名空间)

* 在`.cc` 文件中，允许甚至鼓励使用匿名命名空间，以避免在运行过程中的命名冲突：

```
namespace {                             // .cc 文件中

// 名字空间的内容无需缩进
enum { kUNUSED, kEOF, kERROR };         // 经常使用的符号
bool AtEof() { return pos_ == kEOF; }   // 使用本名字空间内的符号 EOF

} // namespace
```

然而，与特定类关联的文件的作用域声明时在该类中被声明为类型，静态数据成员或静态成员函数，而不是匿名命名空间的成员。如上例所示，匿名空间结束时用注释`//namespace` 标识。

* 不要在 `.h` 文件中使用匿名命名空间。

### 2.1.2 Named Namespaces (具名的命名空间)

具名的命名空间使用方法如下：

* 用命名空间把文件包含，gflags的声明/定义，以及类的前置声明以外的整个源文件封装起来，以区别与其它命名空间：

```
// .h 文件
namespace mynamespace {

// 所有声明都置于命名空间中
// 注意不要使用缩进
class MyClass {
    public:
    …
    void Foo();
};

} // namespace mynamespace
```

```
// .cc 文件
namespace mynamespace {

// 函数定义都置于命名空间中
void MyClass::Foo() {
    …
}

} // namespace mynamespace
```

通常`.cc` 文件包含的更多，更复杂的细节，比如引用其他命名空间的类等。

```
#include “a.h”

DEFINE_bool(someflag, false, “dummy flag”);

class C;                    // 全局名字空间中类 C 的前置声明
namespace a { class A; }    // a::A 的前置声明

namespace b {

…code for b…                // b 中的代码

} // namespace b
```

* 不要在命名空间`std` 内声明任何东西，包括标准库的类前置声明。在`std` 命名空间声明实体会导致不确定的问题，比如不可移植。声明标准库下的实体，需要包含对应的头文件。

* 最好不要使用`using namespace` ,这样可以保证命名空间下的所有名称都可以正常使用。

```
// 禁止--污染名字空间
using namespace foo;
```

* 在`.cc` 文件，`.h`文件的函数，方法或类中，可以使用`using namespace`。

```
// 允许: .cc 文件中
// .h 文件的话, 必须在函数, 方法或类的内部使用
using ::foo::bar;
```

* 在`.cc` 文件，`.h` 文件的函数，方法或类中，允许使用命名空间别称。

```
// 允许: .cc 文件中
// .h 文件的话, 必须在函数, 方法或类的内部使用

namespace fbz = ::foo::bar::baz;

// 在 .h 文件里
namespace librarian {
//以下别名在所有包含了该头文件的文件中生效。
namespace pd_s = ::pipeline_diagnostics::sidetable;

inline void my_inline_function() {
  // namespace alias local to a function (or method).
  namespace fbz = ::foo::bar::baz;
  ...
}
}  // namespace librarian
```

注意在`.h` 文件的别名对包含了该头文件的所有人可见，所以在公共头文件（在项目外可用）以及它们递归包含的其它头文件里，不要用别名。毕竟原则上公共 API 要尽可能地精简。
* 禁止用内联命名空间

## 2.2 Nonmember, Static Member, and Global Functions（非成员函数、静态成员函数和全局函数）
