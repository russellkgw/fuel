# For details on the DSL available within this file, see http://guides.rubyonrails.org/routing.html

Rails.application.routes.draw do
  root 'data_import#index'

  post 'data_import' => 'data_import#create'

  get 'session_controller/create'

end
