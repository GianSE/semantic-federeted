ENTRY
  { author
    booktitle
    howpublished
    institution
    journal
    key
    note
    pages
    publisher
    school
    title
    type
    url
    volume
    year
  }
  {}
  {}

INTEGERS { nameptr numnames sicite.has.real.reference }
STRINGS { s t name }

FUNCTION {not}
{   { #0 }
    { #1 }
  if$
}

FUNCTION {format.names}
{ 's :=
  #1 'nameptr :=
  s num.names$ 'numnames :=
  "" 't :=
  { nameptr numnames > not }
  { s nameptr "{vv~}{ll}{, jj}{, f.}" format.name$ "u" change.case$ 'name :=
    nameptr #1 =
      { name 't := }
      { t "; " * name * 't := }
    if$
    nameptr #1 + 'nameptr :=
  }
  while$
  t
}

FUNCTION {format.authors}
{ author empty$
    { key empty$
        { "Autor(es)" }
        { key }
      if$
    }
    { author format.names }
  if$
}

FUNCTION {write.bold}
{ duplicate$ empty$
    { pop$ }
    { "\textbf{" write$ write$ "}" write$ }
  if$
}

FUNCTION {output.bibitem}
{ newline$
  "\bibitem{" write$
  cite$ write$
  "}" write$
  newline$
}

FUNCTION {write.ref.start}
{ format.authors write$
  ", " write$
  title empty$
    { "titulo" write$ }
    { title write$ }
  if$
}

FUNCTION {write.volume.pages.year}
{ volume empty$
    { skip$ }
    { "v." write$ volume write$ }
  if$
  pages empty$
    { skip$ }
    { volume empty$
        { skip$ }
        { ", " write$ }
      if$
      "p. " write$ pages write$
    }
  if$
  year empty$
    { skip$ }
    { " (" write$ year write$ ")" write$ }
  if$
}

FUNCTION {finish.period}
{ "." write$
  newline$
  newline$
}

FUNCTION {finish.semicolon}
{ ";" write$
  newline$
  newline$
}

FUNCTION {article}
{ output.bibitem
  write.ref.start
  ", " write$
  journal write.bold
  ", " write$
  write.volume.pages.year
  finish.semicolon
}

FUNCTION {book}
{ output.bibitem
  write.ref.start
  ", " write$
  publisher write.bold
  ", " write$
  write.volume.pages.year
  finish.semicolon
}

FUNCTION {mastersthesis}
{ output.bibitem
  write.ref.start
  ", " write$
  type empty$
    { "Dissertacao de Mestrado" write.bold }
    { type write.bold }
  if$
  school empty$
    { skip$ }
    { ", " write$ school write$ }
  if$
  ", " write$
  write.volume.pages.year
  finish.semicolon
}

FUNCTION {phdthesis}
{ output.bibitem
  write.ref.start
  ", " write$
  type empty$
    { "Tese de Doutorado" write.bold }
    { type write.bold }
  if$
  school empty$
    { skip$ }
    { ", " write$ school write$ }
  if$
  ", " write$
  write.volume.pages.year
  finish.period
}

FUNCTION {misc}
{ output.bibitem
  format.authors write$
  ". " write$
  title empty$
    { "Titulo" write$ }
    { title write$ }
  if$
  ". " write$
  howpublished write.bold
  publisher empty$
    { skip$ }
    { ", " write$ publisher write$ }
  if$
  volume empty$
    { skip$ }
    { ", " write$ volume write$ }
  if$
  pages empty$
    { skip$ }
    { ", " write$ pages write$ }
  if$
  year empty$
    { skip$ }
    { " (" write$ year write$ ")" write$ }
  if$
  note empty$
    { skip$ }
    { ". " write$ note write$ }
  if$
  finish.semicolon
}

FUNCTION {manual}
{ output.bibitem
  write.ref.start
  ", " write$
  howpublished write.bold
  url empty$
    { skip$ }
    { ", dispon\'ivel em: " write$ url write$ }
  if$
  note empty$
    { skip$ }
    { ", " write$ note write$ }
  if$
    year empty$
    { skip$ }
    { " (" write$ year write$ ")" write$ }
  if$
  finish.period
}

FUNCTION {inproceedings} { article }
FUNCTION {techreport} { mastersthesis }
FUNCTION {booklet} { manual }
FUNCTION {unpublished} { misc }
FUNCTION {proceedings} { book }
FUNCTION {incollection} { inproceedings }
FUNCTION {default.type} { misc }

FUNCTION {sicite.init.reference.state}
{ #0 'sicite.has.real.reference :=
}

FUNCTION {sicite.check.reference}
{ cite$ #1 #7 substring$ "modelo-" =
    { skip$ }
    { #1 'sicite.has.real.reference := }
  if$
}

FUNCTION {begin.bib}
{ preamble$ empty$
    { skip$ }
    { preamble$ write$ newline$ }
  if$
  sicite.has.real.reference
    { "\sicitereferenciasmodelofalse" write$ newline$ }
    { "\sicitereferenciasmodelotrue" write$ newline$ }
  if$
  "\begin{thebibliography}{99}" write$
  newline$
}

FUNCTION {end.bib}
{ newline$
  "\end{thebibliography}" write$
  newline$
}

READ
EXECUTE {sicite.init.reference.state}
ITERATE {sicite.check.reference}
EXECUTE {begin.bib}
ITERATE {call.type$}
EXECUTE {end.bib}
